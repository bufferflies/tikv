// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::Deref,
    sync::{Arc, Mutex, RwLock},
};

use bytes::Buf;
use cloud::{error::KmsError, KmsProvider};
use dashmap::DashMap;
use derive_more::Deref;
use fxhash::FxHasher;
use hmac::{Hmac, Mac, NewMac};
use kvenginepb::EncryptionMeta;
use openssl::{
    symm,
    symm::{Cipher, Crypter, Mode},
};
use rand::RngCore;
use serde_derive::{Deserialize, Serialize};
use strum::{Display, EnumString};
use thiserror::Error;
use zeroize::Zeroize;

type Hmac256 = Hmac<sha2::Sha256>;
use kvproto::encryptionpb::MasterKeyKms;

// Legacy encryption key format (used in serverless).
// - Exports a 64-byte encryption key.
// - At runtime, this is reduced to a 32-byte data key using HMAC-SHA256.
// - Encoded with data format = 0.
pub const KEY_TYPE_AES_256_CTR_LEGACY: u8 = 1;
const KEY_EXPORTED_SIZE_LEGACY: usize = 1 + 4 + 64;
const ENCRYPTION_DATA_FORMAT_LEGACY: u8 = 0;

// Next-gen key format.
// - Exports a 32-byte data encryption key directly, encrypted by the master
//   key.
// - Encoded with data format = 1.
pub const KEY_TYPE_AES_256_CTR: u8 = 2;
const KEY_EXPORTED_SIZE: usize = 1 + 32;
const ENCRYPTION_DATA_FORMAT: u8 = 1;

pub type Result<T> = std::result::Result<T, EncryptionKeyError>;

#[derive(Debug, Error)]
pub enum EncryptionKeyError {
    #[error("Invalid exported key type {0}")]
    InvalidKeyType(u8),

    #[error("Invalid exported key length {key_len} for type {key_type}")]
    InvalidKeyLength { key_len: usize, key_type: u8 },

    #[error("Data key {0} missing in keyspace {1}")]
    DataKeyMissing(u32, u32),

    #[error("No data key found for shard {0}")]
    NoDataKeyForShard(u64),

    #[error("Unsupported KMS vendor: {0}")]
    UnsupportedVendor(String),

    #[error("Cloud error: {0}")]
    Cloud(#[from] cloud::Error),
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct KeyspaceEncryptionConfig {
    pub enabled: bool,
}
#[derive(Clone)]
pub struct EncryptionKey {
    pub core: Arc<EncryptionKeyCore>,
}

impl std::fmt::Debug for EncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("encryption_key")
            .field(
                "export",
                &self.export().iter().fold(String::new(), |mut acc, byte| {
                    acc.push_str(format!("{:02X}", byte).as_str());
                    acc
                }),
            )
            .finish()
    }
}

impl EncryptionKey {
    pub fn new(cipher_text: Vec<u8>, plain_text: Vec<u8>, key_type: u8) -> Self {
        let plain_text = match key_type {
            KEY_TYPE_AES_256_CTR_LEGACY => {
                EncryptionKeyCore::transform_legacy_encryption_key(&plain_text)
            }
            KEY_TYPE_AES_256_CTR => plain_text,
            _ => unimplemented!(),
        };

        let key_id = key_id_hash_24bits(cipher_text.as_ref());
        Self {
            core: Arc::new(EncryptionKeyCore {
                keyspace_id: None,
                key_id,
                key_type,
                cipher_text,
                plain_text,
                key_manager: None,
            }),
        }
    }

    pub fn with_keyspace_id(&self, keyspace_id: u32) -> Self {
        EncryptionKey {
            core: Arc::new(self.core.with_keyspace_id(keyspace_id)),
        }
    }

    pub fn with_key_manager(&self, m: Arc<EncryptionKeyManager>) -> Self {
        EncryptionKey {
            core: Arc::new(self.core.with_key_manager(m)),
        }
    }

    // Returns the encryption key that matches the given encryption header.
    // Fetch from the key manager if necessary.
    pub fn switch_if_header_mismatch(&self, encryption_header: u32) -> Result<Self> {
        if self.encryption_header() == encryption_header {
            Ok(self.clone())
        } else {
            let m = self.key_manager.clone().unwrap();
            let key = m.get_data_key_by_header(
                self.keyspace_id.expect("keyspace id should be set"),
                encryption_header,
            )?;
            assert_eq!(key.encryption_header(), encryption_header);
            Ok(key.with_key_manager(m))
        }
    }
}

// Computes a non-cryptographic, 3-byte hash from the ciphertext of an
// encryption key, used solely as its ID.
fn key_id_hash_24bits(data: &[u8]) -> [u8; 3] {
    let mut hasher = FxHasher::default();
    hasher.write(data);
    let h = hasher.finish();
    let truncated = (h & 0x00FF_FFFF) as u32;
    [
        ((truncated >> 16) & 0xFF) as u8,
        ((truncated >> 8) & 0xFF) as u8,
        (truncated & 0xFF) as u8,
    ]
}

impl Deref for EncryptionKey {
    type Target = EncryptionKeyCore;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub struct EncryptionKeyCore {
    pub key_type: u8,
    pub key_id: [u8; 3], // three-bytes id
    pub keyspace_id: Option<u32>,
    pub cipher_text: Vec<u8>,
    pub plain_text: Vec<u8>,
    pub key_manager: Option<Arc<EncryptionKeyManager>>,
}

impl EncryptionKeyCore {
    pub fn encrypt(&self, data: &[u8], iv_high: u64, iv_low: u32, buf: &mut Vec<u8>) {
        let mut iv = [0u8; 16];
        iv[0..8].copy_from_slice(&iv_high.to_be_bytes());
        iv[8..12].copy_from_slice(&iv_low.to_be_bytes());
        let cipher = Cipher::aes_256_ctr();
        let mut crypter = Crypter::new(cipher, Mode::Encrypt, &self.plain_text, Some(&iv)).unwrap();
        let origin_len = buf.len();
        let additional_len = data.len() + cipher.block_size();
        buf.resize(origin_len + additional_len, 0);
        let dst = &mut buf[origin_len..];
        let count = crypter.update(data, dst).unwrap();
        let rest = crypter.finalize(&mut dst[count..]).unwrap();
        buf.truncate(origin_len + count + rest);
    }

    pub fn decrypt(
        &self,
        ciphertext: &[u8],
        iv_high: u64,
        iv_low: u32,
        _encryption_ver: u32,
        buf: &mut Vec<u8>,
    ) {
        let mut iv = [0u8; 16];
        iv[0..8].copy_from_slice(&iv_high.to_be_bytes());
        iv[8..12].copy_from_slice(&iv_low.to_be_bytes());
        let cipher = Cipher::aes_256_ctr();
        let mut crypter = Crypter::new(cipher, Mode::Decrypt, &self.plain_text, Some(&iv)).unwrap();
        let origin_len = buf.len();
        let additional_len = ciphertext.len() + cipher.block_size();
        buf.resize(origin_len + additional_len, 0);
        let dst = &mut buf[origin_len..];
        let count = crypter.update(ciphertext, dst).unwrap();
        let rest = crypter.finalize(&mut dst[count..]).unwrap();
        buf.truncate(origin_len + count + rest);
    }

    pub fn export(&self) -> Vec<u8> {
        match self.key_type {
            KEY_TYPE_AES_256_CTR_LEGACY => {
                let mut data = Vec::with_capacity(KEY_EXPORTED_SIZE_LEGACY);
                let current_ver: u32 = 0;
                data.push(KEY_TYPE_AES_256_CTR_LEGACY);
                data.extend_from_slice(&current_ver.to_be_bytes());
                data.extend_from_slice(&self.cipher_text);
                data
            }
            KEY_TYPE_AES_256_CTR => {
                let mut data = Vec::with_capacity(KEY_EXPORTED_SIZE);
                data.push(KEY_TYPE_AES_256_CTR);
                data.extend_from_slice(&self.cipher_text);
                data
            }
            _ => unreachable!(),
        }
    }

    // Transforms a legacy 64-byte encryption key into a 32-byte data key using
    // HMAC-SHA256 with a fixed version.
    //
    // V1 always uses version 0 since key rotation was not supported.
    fn transform_legacy_encryption_key(plain_text: &[u8]) -> Vec<u8> {
        let key_version: u32 = 0;
        let mut hmac = Hmac256::new_varkey(plain_text).unwrap();
        hmac.update(&key_version.to_be_bytes());
        hmac.finalize().into_bytes().to_vec()
    }

    pub fn get_key_id(&self) -> u32 {
        (self.key_id[0] as u32) << 16 | (self.key_id[1] as u32) << 8 | (self.key_id[2] as u32)
    }

    pub fn get_type(&self) -> u8 {
        self.key_type
    }

    pub fn with_keyspace_id(&self, keyspace_id: u32) -> Self {
        Self {
            key_type: self.key_type,
            key_id: self.key_id,
            keyspace_id: Some(keyspace_id),
            cipher_text: self.cipher_text.clone(),
            plain_text: self.plain_text.clone(),
            key_manager: self.key_manager.clone(),
        }
    }

    pub fn with_key_manager(&self, key_manager: Arc<EncryptionKeyManager>) -> Self {
        Self {
            key_type: self.key_type,
            key_id: self.key_id,
            keyspace_id: self.keyspace_id,
            cipher_text: self.cipher_text.clone(),
            plain_text: self.plain_text.clone(),
            key_manager: Some(key_manager),
        }
    }

    /// Returns the 4-byte encryption header that identifies the data key within
    /// the keyspace. The first byte is the data format version; the remaining 3
    /// bytes encode the data key ID.
    ///
    /// This header is written alongside the encrypted data and is required
    /// later to locate and use the correct data key during decryption.
    pub fn encryption_header(&self) -> u32 {
        match self.key_type {
            KEY_TYPE_AES_256_CTR_LEGACY => ENCRYPTION_DATA_FORMAT_LEGACY as u32,
            KEY_TYPE_AES_256_CTR => ((ENCRYPTION_DATA_FORMAT as u32) << 24) | self.get_key_id(),
            _ => unreachable!(),
        }
    }

    pub fn encryption_block_size(&self) -> usize {
        Cipher::aes_256_ctr().block_size()
    }
}

impl Drop for EncryptionKeyCore {
    fn drop(&mut self) {
        // Zeroize sensitive fields.
        self.plain_text.zeroize();
    }
}

#[derive(Debug, EnumString, Display, PartialEq)]
#[strum(serialize_all = "kebab_case", ascii_case_insensitive)]
pub enum KmsVendor {
    Aws,
    Aliyun,
    Test,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct MasterKeyConfig {
    // Encrypted master key, base64 encoded.
    pub cipher_text: String,
    // KMS info
    pub key_id: String,
    pub region: String,
    pub vendor: String,
    pub endpoint: String,
}

impl MasterKeyConfig {
    pub fn override_from_env(&mut self) {
        Self::env_or_default("CSE_MASTER_KEY_ID", &mut self.key_id);
        Self::env_or_default("CSE_MASTER_KEY_CIPHER_TEXT", &mut self.cipher_text);
        Self::env_or_default("CSE_MASTER_KEY_VENDOR", &mut self.vendor);
        Self::env_or_default("AWS_REGION", &mut self.region);
    }

    fn env_or_default(name: &str, val: &mut String) {
        if let Ok(v) = std::env::var(name) {
            *val = v;
        }
    }

    /// Generates a new master key and its encrypted form using the configured
    /// KMS vendor.
    pub async fn generate_new_master_key(&self) -> Result<(MasterKey, cloud::EncryptedKey)> {
        let vendor = self.vendor.parse::<KmsVendor>();
        match vendor {
            Ok(KmsVendor::Aws) => {
                let mut kms_cfg = MasterKeyKms::default();
                kms_cfg.key_id = self.key_id.clone();
                kms_cfg.vendor = self.vendor.clone();
                kms_cfg.endpoint = self.endpoint.clone();
                kms_cfg.region = self.region.clone();

                let conf = cloud::kms::Config::from_proto(kms_cfg).unwrap();
                let kms = aws::AwsKms::new(conf).unwrap();
                let key_pair = kms.generate_data_key().await?;

                let master_key = MasterKey::new(key_pair.plaintext.as_slice());
                let master_key_ciphertext =
                    cloud::EncryptedKey::new(key_pair.encrypted.into_inner()).unwrap();
                Ok((master_key, master_key_ciphertext))
            }
            Ok(KmsVendor::Aliyun) => {
                // TODO: support Aliyun master key generation.
                Err(EncryptionKeyError::UnsupportedVendor(self.vendor.clone()))
            }
            Ok(KmsVendor::Test) => {
                let mut master_key = self.key_id.as_bytes().to_vec();
                master_key.resize(32, 11);
                let master_key_ciphertext = cloud::EncryptedKey::new(master_key.clone()).unwrap(); // no encryption
                Ok((MasterKey::new(&master_key), master_key_ciphertext))
            }
            _ => Err(EncryptionKeyError::UnsupportedVendor(self.vendor.clone())),
        }
    }

    /// Decrypts the encrypted master key using the configured KMS vendor.
    pub async fn decrypt(&self) -> Result<MasterKey> {
        let vendor = self.vendor.parse::<KmsVendor>();
        match vendor {
            Ok(KmsVendor::Aws) => {
                let mut kms_cfg = MasterKeyKms::default();
                kms_cfg.key_id = self.key_id.clone();
                kms_cfg.vendor = self.vendor.clone();
                kms_cfg.endpoint = self.endpoint.clone();
                kms_cfg.region = self.region.clone();
                let conf = cloud::kms::Config::from_proto(kms_cfg).unwrap();
                let aws_kms = aws::AwsKms::new(conf).unwrap();
                let encrypted_key =
                    cloud::EncryptedKey::new(base64::decode(&self.cipher_text).unwrap()).unwrap();
                let master_key_plain_text = aws_kms.decrypt_data_key(&encrypted_key).await?;
                Ok(MasterKey::new(&master_key_plain_text))
            }
            Ok(KmsVendor::Aliyun) => {
                let master_key_plain_text = aliyun::decrypt_master_key(&self.cipher_text)
                    .await
                    .map_err(|e| cloud::Error::KmsError(KmsError::Other(Box::new(e))))?;
                Ok(MasterKey::new(&master_key_plain_text))
            }
            Ok(KmsVendor::Test) => {
                let mut master_key = self.key_id.as_bytes().to_vec();
                master_key.resize(32, 11);
                Ok(MasterKey::new(&master_key))
            }
            _ => {
                // use a fixed master key for test.
                let master_key = vec![1u8; 32];
                Ok(MasterKey::new(&master_key))
            }
        }
    }
}

#[derive(Clone)]
pub struct MasterKey {
    core: Arc<MasterKeyCore>,
}

impl MasterKey {
    pub fn new(plain_text: &[u8]) -> Self {
        Self {
            core: Arc::new(MasterKeyCore {
                master_key: plain_text.to_vec(),
            }),
        }
    }
}

impl Deref for MasterKey {
    type Target = MasterKeyCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

// ExportedMasterKey is a dedicated type for exported master key,
// which is used to avoid accidentally printing the master key.
#[derive(Deref, PartialEq)]
pub struct ExportedMasterKey(Vec<u8>);

impl ExportedMasterKey {
    pub fn new(master_key: Vec<u8>) -> Self {
        Self(master_key)
    }
}

impl std::fmt::Debug for ExportedMasterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ExportedMasterKey")
            .field(&"REDACTED".to_string())
            .finish()
    }
}

pub struct MasterKeyCore {
    master_key: Vec<u8>,
}

impl MasterKeyCore {
    pub fn generate_encryption_key(&self) -> EncryptionKey {
        let mut rng = rand::thread_rng();
        let mut plain_text = vec![0u8; 32];
        rng.fill_bytes(&mut plain_text[..]);
        let cipher_text =
            symm::encrypt(Cipher::aes_256_ctr(), &self.master_key, None, &plain_text).unwrap();
        EncryptionKey::new(cipher_text, plain_text, KEY_TYPE_AES_256_CTR)
    }

    pub fn decrypt_encryption_key(&self, mut exported: &[u8]) -> Result<EncryptionKey> {
        let key_len = exported.len();
        let key_type = exported.get_u8();
        let expected_len = match key_type {
            KEY_TYPE_AES_256_CTR_LEGACY => KEY_EXPORTED_SIZE_LEGACY,
            KEY_TYPE_AES_256_CTR => KEY_EXPORTED_SIZE,
            _ => return Err(EncryptionKeyError::InvalidKeyType(key_type)),
        };
        if expected_len != key_len {
            return Err(EncryptionKeyError::InvalidKeyLength { key_len, key_type });
        }
        if key_type == KEY_TYPE_AES_256_CTR_LEGACY {
            let key_ver = exported.get_u32();
            assert!(key_ver == 0);
        }
        let cipher_text = exported.to_vec();
        let plain_text =
            symm::decrypt(Cipher::aes_256_ctr(), &self.master_key, None, &cipher_text).unwrap();
        Ok(EncryptionKey::new(cipher_text, plain_text, key_type))
    }

    pub fn export(&self) -> ExportedMasterKey {
        ExportedMasterKey::new(self.master_key.clone())
    }
}

impl Drop for MasterKeyCore {
    fn drop(&mut self) {
        // Zeroize sensitive fields.
        self.master_key.zeroize();
    }
}

use std::hash::{Hash, Hasher};

use protobuf::Message;
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EncryptionKeyRef {
    keyspace_id: u32,
    data_key_id: u32,
}

/// Extracts the current data key from serialized `EncryptionMeta` bytes.
///
/// This is typically used on the write path to ensure the correct data key is
/// used for encryption. Internally registers the `EncryptionMeta` if needed and
/// returns the active data key associated with it.
pub fn current_data_key_from_encryption_meta_bytes(
    m: Arc<EncryptionKeyManager>,
    encryption_meta_bytes: &[u8],
) -> Result<EncryptionKey> {
    let key = m.register_encryption_meta_if_needed(encryption_meta_bytes);
    key.map(|k| k.with_key_manager(m))
}

/// A singleton that manages encryption keys across all keyspaces.
pub struct EncryptionKeyManager {
    // Tokio runtime for handling KMS operations (master key decryption).
    runtime: Arc<tokio::runtime::Runtime>,

    // shard_id -> (keyspace_id, data_key_id)
    shard_to_current_key: DashMap<u64, EncryptionKeyRef>,

    // keyspace_id -> keyspace encryption entry
    keyspace_to_encryption_entry: DashMap<u32, Arc<KeyspaceEncryptionEntry>>,
}

impl Default for EncryptionKeyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EncryptionKeyManager {
    pub fn new() -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("encryption-manager")
                .enable_all()
                .build()
                .unwrap(),
        );

        Self {
            runtime,
            shard_to_current_key: DashMap::new(),
            keyspace_to_encryption_entry: DashMap::new(),
        }
    }

    /// Registers the given `EncryptionMeta` if it's newer than the cached
    /// version, and returns the current data key.
    fn register_encryption_meta_if_needed(
        &self,
        encryption_meta_bytes: &[u8],
    ) -> Result<EncryptionKey> {
        let mut incoming = kvenginepb::EncryptionMeta::default();
        incoming.merge_from_bytes(encryption_meta_bytes).unwrap();

        let entry = self
            .keyspace_to_encryption_entry
            .entry(incoming.keyspace_id)
            .or_insert_with(|| {
                Arc::new(KeyspaceEncryptionEntry::new(
                    &incoming,
                    self.runtime.clone(),
                ))
            })
            .clone();
        entry.update_meta_if_newer(&incoming);

        let data_key = entry.get_data_key(incoming.get_current().data_key_id)?;
        Ok(data_key)
    }

    /// Registers the current encryption key used by a shard (region).
    pub fn register_current_key_for_shard(&self, shard_id: u64, k: EncryptionKey) {
        if k.get_type() == KEY_TYPE_AES_256_CTR_LEGACY {
            // We don't expect the EncryptionKeyManager to deal with
            // legacy keys.
            return;
        }
        assert!(k.keyspace_id.is_some());
        let keyspace_id = k.keyspace_id.unwrap();
        let key_ref = EncryptionKeyRef {
            keyspace_id,
            data_key_id: k.get_key_id(),
        };
        self.shard_to_current_key.insert(shard_id, key_ref);
    }

    /// Returns the current encryption key for the given shard, if available.
    ///
    /// Might be used by TiFlash FFI.
    pub fn current_key_by_shard(&self, shard_id: u64) -> Result<EncryptionKey> {
        let key_ref = self
            .shard_to_current_key
            .get(&shard_id)
            .ok_or_else(|| EncryptionKeyError::NoDataKeyForShard(shard_id))?;
        let entry = self
            .keyspace_to_encryption_entry
            .get(&key_ref.keyspace_id)
            .unwrap();
        entry.get_data_key(key_ref.data_key_id)
    }

    /// Deregisters the encryption key associated with a shard.
    pub fn deregister_current_key_for_shard(&self, shard_id: u64) {
        self.shard_to_current_key.remove(&shard_id);
    }

    /// Retrieves the data key for a given keyspace and encryption header.
    fn get_data_key_by_header(
        &self,
        keyspace_id: u32,
        encryption_header: u32,
    ) -> Result<EncryptionKey> {
        let encryption_format = (encryption_header >> 24) as u8;
        match encryption_format {
            ENCRYPTION_DATA_FORMAT_LEGACY => {
                // We don't expect the encryption key manager to manage
                // V1 format keys.
                panic!("unexpected V1 encryption key format");
            }
            ENCRYPTION_DATA_FORMAT => {
                let data_key_id = encryption_header & 0x00FF_FFFF;
                let entry = self.keyspace_to_encryption_entry.get(&keyspace_id).unwrap();
                entry.get_data_key(data_key_id)
            }
            _ => unreachable!(),
        }
    }
}

/// Holds all encryption-related info for a single keyspace.
///
/// It handles the caching and lookup of encryption metadata, data keys,
/// and master keys.
struct KeyspaceEncryptionEntry {
    // Tokio runtime for handling KMS operations (master key decryption).
    runtime: Arc<tokio::runtime::Runtime>,

    // meta is read-often, updated on rotation
    meta: RwLock<Arc<EncryptionMeta>>,

    // data_key_id -> key (fast reads)
    data_key_cache: DashMap<u32, EncryptionKey>,

    // ciphertext -> decrypted master key
    master_key_cache: DashMap<Vec<u8>, Arc<std::sync::Mutex<Option<Arc<MasterKey>>>>>,
}

impl KeyspaceEncryptionEntry {
    pub fn new(encryption_meta: &EncryptionMeta, rt: Arc<tokio::runtime::Runtime>) -> Self {
        Self {
            meta: RwLock::new(Arc::new(encryption_meta.clone())),
            data_key_cache: DashMap::new(),
            master_key_cache: DashMap::new(),
            runtime: rt,
        }
    }

    pub fn update_meta_if_newer(&self, incoming: &kvenginepb::EncryptionMeta) {
        let mut g = self.meta.write().unwrap();
        let old_file_id = g.current.as_ref().map(|e| e.file_id).unwrap_or(0);
        let new_file_id = incoming.current.as_ref().map(|e| e.file_id).unwrap_or(0);
        if new_file_id > old_file_id {
            *g = Arc::new(incoming.clone());
        }
    }

    /// Retrieves or decrypts the data key from metadata and caches it.
    fn get_data_key(&self, data_key_id: u32) -> Result<EncryptionKey> {
        if let Some(key) = self.data_key_cache.get(&data_key_id) {
            return Ok(key.value().clone());
        }

        let m = self.meta.read().unwrap();
        let data_key = m
            .data_keys
            .get(&data_key_id)
            .ok_or_else(|| EncryptionKeyError::DataKeyMissing(data_key_id, m.keyspace_id))?;

        let master_key = self.decrypt_master_key(&m)?;
        master_key
            .decrypt_encryption_key(&data_key.ciphertext)
            .map(|key| {
                let key = key.with_keyspace_id(m.keyspace_id);
                self.data_key_cache.insert(data_key_id, key.clone());
                key
            })
    }

    /// Decrypts the master key from the given `EncryptionMeta`.
    ///
    /// If the decrypted master key is already cached, it is returned directly.
    /// Otherwise, it is decrypted via KMS and stored for future reuse.
    pub fn decrypt_master_key(&self, m: &EncryptionMeta) -> Result<Arc<MasterKey>> {
        let master_key = m.master_key.clone().unwrap();
        let ciphertext = master_key.ciphertext.clone();

        // Insert a shared mutex-protected slot for this master key if it
        // doesn't exist. The mutex is for avoiding duplicate calls to KMS.
        let entry = self
            .master_key_cache
            .entry(ciphertext)
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone();

        let mut guard = entry.lock().unwrap();
        if let Some(master_key) = &*guard {
            return Ok(master_key.clone());
        }

        let master_key_config = MasterKeyConfig {
            key_id: master_key.cmek_id.clone(),
            vendor: master_key.vendor.clone(),
            region: master_key.region.clone(),
            endpoint: master_key.endpoint.clone(),
            cipher_text: base64::encode(master_key.ciphertext.clone()),
        };
        let fut = master_key_config.decrypt();
        let master_key = if tokio::runtime::Handle::try_current().is_ok() {
            // Inside a Tokio runtime (e.g. in
            // `TxnChunkHandler::acquire_keyspace_info`), calling
            // `block_on` directly would stall the scheduler. Use
            // `block_in_place` instead.
            tokio::task::block_in_place(|| self.runtime.block_on(fut))
        } else {
            self.runtime.block_on(fut)
        }?;
        let arc_key = Arc::new(master_key);
        *guard = Some(arc_key.clone());
        Ok(arc_key)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    pub fn generate_encryption_key_v1(master_key: &MasterKey) -> EncryptionKey {
        let mut rng = rand::thread_rng();
        let mut plain_text = vec![0u8; 64];
        rng.fill_bytes(&mut plain_text[..]);
        let cipher_text = symm::encrypt(
            Cipher::aes_256_ctr(),
            &master_key.export(),
            None,
            &plain_text,
        )
        .unwrap();
        EncryptionKey::new(cipher_text, plain_text, KEY_TYPE_AES_256_CTR_LEGACY)
    }

    #[test]
    fn test_kms_vendor_parse() {
        let cases = [
            ("aws", KmsVendor::Aws),
            ("Aws", KmsVendor::Aws),
            ("AWS", KmsVendor::Aws),
            ("aliyun", KmsVendor::Aliyun),
            ("Aliyun", KmsVendor::Aliyun),
            ("test", KmsVendor::Test),
        ];

        for (input, expected) in cases {
            assert_eq!(expected, input.parse::<KmsVendor>().unwrap());
        }
    }
    #[test]
    fn test_encryption() {
        let master_key_plain_text = rand::thread_rng().gen::<[u8; 32]>().to_vec();
        let master_key = MasterKey::new(&master_key_plain_text);
        for encryption_key in [
            generate_encryption_key_v1(&master_key),
            master_key.generate_encryption_key(),
        ] {
            let mut encrypted = vec![];
            encryption_key.encrypt(b"hello", 123, 12, &mut encrypted);

            // Test the key export code path.
            let exported = encryption_key.export();
            let encryption_key = master_key
                .decrypt_encryption_key(&exported)
                .expect("failed to decrypt encryption key");

            let mut decrypted = vec![];
            encryption_key.decrypt(&encrypted, 123, 12, 0, &mut decrypted);
            assert_eq!(decrypted, b"hello");
        }
    }

    fn new_test_runtime() -> Arc<tokio::runtime::Runtime> {
        Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_master_key_gen() {
        let mut config = MasterKeyConfig::default();
        config.vendor = "test".to_owned();
        let runtime = new_test_runtime();
        let (master_key, encrypted_key) =
            runtime.block_on(config.generate_new_master_key()).unwrap();
        config.cipher_text = base64::encode(encrypted_key.to_vec());
        let decoded_master_key = runtime.block_on(config.decrypt()).unwrap();
        assert_eq!(master_key.export(), decoded_master_key.export());
    }

    #[test]
    fn test_encryption_key_manager() {
        let mut config = MasterKeyConfig::default();
        config.vendor = "test".to_owned();
        let (master_key, master_key_ciphertext) = new_test_runtime()
            .block_on(config.generate_new_master_key())
            .unwrap();

        let data_key1 = master_key.generate_encryption_key();
        let data_key2 = master_key.generate_encryption_key();
        let keyspace_id = 100;

        let mut m = kvenginepb::EncryptionMeta::default();
        m.set_keyspace_id(keyspace_id);

        let mut mk = kvenginepb::MasterKey::default();
        mk.vendor = config.vendor.clone();
        mk.ciphertext = master_key_ciphertext.to_vec();
        m.set_master_key(mk);

        m.mut_current().data_key_id = data_key2.get_key_id();
        for k in [&data_key1, &data_key2] {
            let mut dk = kvenginepb::DataKey::new();
            dk.set_ciphertext(k.export());
            m.mut_data_keys().insert(k.get_key_id(), dk);
        }

        let mgr = Arc::new(EncryptionKeyManager::new());
        let k =
            current_data_key_from_encryption_meta_bytes(mgr.clone(), &m.write_to_bytes().unwrap())
                .expect("failed to get current key");
        assert_eq!(k.core.plain_text, data_key2.core.plain_text);
        for dk in [&data_key1, &data_key2] {
            assert_eq!(
                k.switch_if_header_mismatch(dk.encryption_header())
                    .unwrap()
                    .core
                    .plain_text,
                dk.core.plain_text
            )
        }

        let err = k
            .switch_if_header_mismatch(master_key.generate_encryption_key().encryption_header())
            .unwrap_err();
        assert!(matches!(err, EncryptionKeyError::DataKeyMissing(_, _)));

        // Ensure that `keyspace_id` is set on `data_key1`.
        let data_key1 = k
            .switch_if_header_mismatch(data_key1.encryption_header())
            .unwrap();
        assert!(matches!(
            mgr.current_key_by_shard(5),
            Err(EncryptionKeyError::NoDataKeyForShard(_))
        ));
        mgr.register_current_key_for_shard(5, data_key1.clone());
        mgr.current_key_by_shard(5).unwrap();
        assert_eq!(
            mgr.current_key_by_shard(5).unwrap().core.plain_text,
            data_key1.core.plain_text
        );
        mgr.deregister_current_key_for_shard(5);
        assert!(matches!(
            mgr.current_key_by_shard(5),
            Err(EncryptionKeyError::NoDataKeyForShard(_))
        ));
    }
}

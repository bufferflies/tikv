// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

// TODO(iosmanthus): introduce TLS support for the whole integration test.

use std::{convert::TryFrom, error::Error, fs, io, iter};

use hyper::{client::HttpConnector, server::conn::AddrIncoming, Uri};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder, TlsAcceptor};
use rustls::server::AllowAnyAuthenticatedClient;
use rustls_pemfile::Item;
use tikv_util::Either;

use crate::SecurityManager;

pub type Result<T> = std::result::Result<T, Box<dyn Error + Sync + Send>>;

impl SecurityManager {
    pub fn acceptor(&self, incoming: AddrIncoming) -> Result<Either<AddrIncoming, TlsAcceptor>> {
        if self.cfg.ca_path.is_empty() {
            Ok(Either::Left(incoming))
        } else {
            Ok(Either::Right(self.tls_acceptor(incoming)?))
        }
    }

    pub fn build_uri<T: AsRef<str>>(&self, rest: T) -> Result<Uri> {
        let scheme = if self.cfg.ca_path.is_empty() {
            "http"
        } else {
            "https"
        };
        Ok(Uri::try_from(format!("{}://{}", scheme, rest.as_ref()))?)
    }

    fn tls_acceptor(&self, incoming: AddrIncoming) -> Result<TlsAcceptor> {
        let ca = load_root_store(&self.cfg.ca_path)?;
        let cert = load_certs(&self.cfg.cert_path)?;
        let key = load_key(&self.cfg.key_path)?;

        let tls_config = rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_client_cert_verifier(AllowAnyAuthenticatedClient::new(ca).boxed())
            .with_single_cert(cert, key)?;
        let acceptor = TlsAcceptor::builder()
            .with_tls_config(tls_config)
            .with_all_versions_alpn()
            .with_incoming(incoming);

        Ok(acceptor)
    }

    pub fn http_client(
        &self,
        client_builder: hyper::client::Builder,
    ) -> Result<hyper::Client<HttpsConnector<HttpConnector>>> {
        if self.cfg.ca_path.is_empty() {
            Ok(client_builder.build(
                HttpsConnectorBuilder::new()
                    .with_native_roots()
                    .https_or_http()
                    .enable_http1()
                    .build(),
            ))
        } else {
            let ca = load_root_store(&self.cfg.ca_path)?;
            let cert = load_certs(&self.cfg.cert_path)?;
            let key = load_key(&self.cfg.key_path)?;
            let tls_config = rustls::ClientConfig::builder()
                .with_safe_defaults()
                .with_root_certificates(ca)
                .with_client_auth_cert(cert, key)?;

            let connector = hyper_rustls::HttpsConnectorBuilder::new()
                .with_tls_config(tls_config)
                .https_only()
                .enable_http1()
                .build();
            Ok(client_builder.build(connector))
        }
    }
}

fn error<E>(err: E) -> io::Error
where
    E: Into<Box<dyn Error + Send + Sync>>,
{
    io::Error::new(io::ErrorKind::Other, err)
}

fn load_certs(filename: &str) -> io::Result<Vec<rustls::Certificate>> {
    let certfile = fs::File::open(filename)?;
    let mut reader = io::BufReader::new(certfile);
    let certs = rustls_pemfile::certs(&mut reader)?;
    Ok(certs.into_iter().map(rustls::Certificate).collect())
}

fn load_key(filename: &str) -> io::Result<rustls::PrivateKey> {
    let keyfile = fs::File::open(filename)?;
    let mut reader = io::BufReader::new(keyfile);
    if let Some(item) = iter::from_fn(|| rustls_pemfile::read_one(&mut reader).transpose()).next() {
        let key = match item? {
            Item::RSAKey(key) => key,
            Item::PKCS8Key(key) => key,
            Item::ECKey(key) => key,
            _ => return Err(error("unsupported private key type")),
        };
        return Ok(rustls::PrivateKey(key));
    }
    Err(error("no private keys found"))
}

fn load_root_store(filename: &str) -> io::Result<rustls::RootCertStore> {
    let ca_certs = load_certs(filename)?;
    let mut store = rustls::RootCertStore::empty();
    for cert in ca_certs.iter() {
        store.add(cert).map_err(error)?;
    }
    Ok(store)
}

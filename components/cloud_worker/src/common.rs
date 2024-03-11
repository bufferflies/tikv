// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{borrow::Cow, collections::HashMap, str::FromStr};

use futures::{future::ok, TryStreamExt};
use http::{header, Response, StatusCode};
use hyper::Body;

pub(crate) fn get_param<T: FromStr>(
    query_pairs: &HashMap<Cow<'_, str>, Cow<'_, str>>,
    name: &str,
) -> Option<T> {
    query_pairs.get(name).and_then(|x| T::from_str(x).ok())
}

pub(crate) async fn get_body(req: hyper::Request<hyper::Body>) -> hyper::Result<Vec<u8>> {
    let length = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .map(|x| usize::from_str(x.to_str().unwrap_or_default()).unwrap_or_default())
        .unwrap_or_default();
    let mut body = Vec::with_capacity(length);
    req.into_body()
        .try_for_each(|bytes| {
            body.extend(bytes);
            ok(())
        })
        .await?;
    Ok(body)
}

pub(crate) fn make_response<T>(status_code: StatusCode, message: T) -> Response<Body>
where
    T: Into<Body>,
{
    Response::builder()
        .status(status_code)
        .body(message.into())
        .unwrap()
}

pub(crate) fn make_json_response<T>(status_code: StatusCode, resp: &T) -> Response<Body>
where
    T: ?Sized + serde::Serialize,
{
    let json = serde_json::to_string(resp).unwrap();
    Response::builder()
        .status(status_code)
        .header(header::CONTENT_TYPE, "application/json")
        .body(json.into())
        .unwrap()
}

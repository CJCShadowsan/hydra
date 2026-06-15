use std::net::TcpStream;

use anyhow::{Result, bail};
use skippy_protocol::binary::{WireMessageKind, recv_reply};

#[derive(Default)]
pub struct PredictionReturnHub;

impl PredictionReturnHub {
    pub(crate) fn handle_return_connection(
        &self,
        open: skippy_protocol::binary::StageWireMessage,
        mut stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        while recv_reply(&mut stream).is_ok() {}
        Ok(())
    }
}

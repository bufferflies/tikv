// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use rand::Rng as _;

#[derive(Debug, Copy, Clone)]
pub struct MaxAttemptsExceededError;

impl std::error::Error for MaxAttemptsExceededError {
    fn description(&self) -> &str {
        "max retry attempts exceeded"
    }
}

impl std::fmt::Display for MaxAttemptsExceededError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "max retry attempts exceeded")
    }
}

#[derive(Debug)]
pub struct ExponentialBackoff {
    current_delay_ms: u64,
    max_delay_ms: u64,
    max_attempts: usize,
    current_attempts: usize,
}

impl ExponentialBackoff {
    /// Creates a new instance of `ExponentialBackoff`.
    ///
    /// # Arguments
    /// * `base_delay` - The initial delay duration.
    /// * `max_delay` - The maximum delay duration.
    /// * `max_attempts` - The maximum number of attempts before giving up.
    pub fn new(base_delay: Duration, max_delay: Duration, max_attempts: usize) -> Self {
        ExponentialBackoff {
            current_delay_ms: base_delay.as_millis() as u64,
            max_delay_ms: max_delay.as_millis() as u64,
            max_attempts,
            current_attempts: 0,
        }
    }

    pub fn next_delay(&mut self) -> Result<Duration, MaxAttemptsExceededError> {
        if self.current_attempts >= self.max_attempts {
            return Err(MaxAttemptsExceededError);
        }

        self.current_attempts += 1;
        self.current_delay_ms = std::cmp::min(self.current_delay_ms << 1, self.max_delay_ms);
        let delay =
            rand::thread_rng().gen_range(self.current_delay_ms >> 1..=self.current_delay_ms);
        Ok(Duration::from_millis(delay))
    }

    pub fn current_attempts(&self) -> usize {
        self.current_attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff() {
        let base_delay = Duration::from_millis(100);
        let max_delay = Duration::from_secs(1);
        let max_attempts = 5;

        let mut backoff = ExponentialBackoff::new(base_delay, max_delay, max_attempts);

        let mut delay = backoff.next_delay().unwrap().as_millis();
        assert!((100..=200).contains(&delay));
        delay = backoff.next_delay().unwrap().as_millis();
        assert!((200..=400).contains(&delay));
        delay = backoff.next_delay().unwrap().as_millis();
        assert!((400..=800).contains(&delay));
        delay = backoff.next_delay().unwrap().as_millis();
        assert!((500..=1000).contains(&delay));
        delay = backoff.next_delay().unwrap().as_millis();
        assert!((500..=1000).contains(&delay));

        backoff.next_delay().unwrap_err();
    }
}

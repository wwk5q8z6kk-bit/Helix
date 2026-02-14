use anyhow::Result;
use hx_core::credentials::CredentialStore;
use std::io::{self, BufRead};

fn store() -> CredentialStore {
    CredentialStore::new("helix")
}

/// Convert a hostname to a Helix secret key.
///
/// Convention: `GIT_CREDENTIAL_{HOST}` where dots become underscores and the name is uppercased.
/// Example: `github.com` -> `GIT_CREDENTIAL_GITHUB_COM`
fn host_to_secret_key(host: &str) -> String {
    let normalized = host
        .to_ascii_uppercase()
        .replace('.', "_")
        .replace('-', "_");
    format!("GIT_CREDENTIAL_{normalized}")
}

/// Parse the git credential protocol input from stdin.
///
/// Git sends key=value pairs, one per line, terminated by an empty line.
/// We extract `protocol`, `host`, and `username` if present.
fn parse_credential_input() -> Result<(Option<String>, Option<String>, Option<String>)> {
    let stdin = io::stdin();
    let mut protocol = None;
    let mut host = None;
    let mut username = None;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once('=') {
            match key {
                "protocol" => protocol = Some(value.to_string()),
                "host" => host = Some(value.to_string()),
                "username" => username = Some(value.to_string()),
                _ => {}
            }
        }
    }

    Ok((protocol, host, username))
}

/// `mv git-credential get` — implements the git credential helper "get" action.
///
/// Reads protocol/host from stdin, looks up the matching secret, and outputs
/// the credential in git's expected format.
pub async fn get() -> Result<()> {
    let (protocol, host, _username) = parse_credential_input()?;

    let host = match host {
        Some(h) => h,
        None => {
            // No host provided — nothing to look up
            return Ok(());
        }
    };

    let secret_key = host_to_secret_key(&host);
    let creds = store();

    match creds.get(&secret_key)? {
        Some(sv) => {
            let password = sv.expose();

            // Output in git credential protocol format
            if let Some(proto) = protocol {
                println!("protocol={proto}");
            }
            println!("host={host}");
            println!("username=x-access-token");
            println!("password={password}");
            println!();
        }
        None => {
            // Secret not found — git will fall through to the next helper
        }
    }

    Ok(())
}

/// `mv git-credential install` — configures git to use Helix as a credential helper.
pub async fn install() -> Result<()> {
    // Find the mv binary path
    let hx_path = std::env::current_exe()?;
    let hx_path_str = hx_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("binary path contains non-UTF8 characters"))?;

    let helper_value = format!("!{hx_path_str} git-credential");

    let output = tokio::process::Command::new("git")
        .args(["config", "--global", "credential.helper", &helper_value])
        .output()
        .await?;

    if output.status.success() {
        println!("git credential helper installed globally:");
        println!("  credential.helper = {helper_value}");
        println!();
        println!("Store credentials with:");
        println!("  mv secret set GIT_CREDENTIAL_GITHUB_COM <your-token>");
        println!();
        println!("Convention: GIT_CREDENTIAL_<HOST> with dots as underscores, uppercased.");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git config failed: {stderr}");
    }

    Ok(())
}

/// `mv git-credential uninstall` — removes the Helix git credential helper.
pub async fn uninstall() -> Result<()> {
    let output = tokio::process::Command::new("git")
        .args(["config", "--global", "--unset-all", "credential.helper"])
        .output()
        .await?;

    if output.status.success() || output.status.code() == Some(5) {
        // Exit code 5 means the key didn't exist — still success for uninstall
        println!("git credential helper removed from global config");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git config --unset-all failed: {stderr}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_to_key_converts_github() {
        assert_eq!(
            host_to_secret_key("github.com"),
            "GIT_CREDENTIAL_GITHUB_COM"
        );
    }

    #[test]
    fn host_to_key_converts_gitlab_with_port() {
        assert_eq!(
            host_to_secret_key("gitlab.example.com"),
            "GIT_CREDENTIAL_GITLAB_EXAMPLE_COM"
        );
    }

    #[test]
    fn host_to_key_handles_dashes() {
        assert_eq!(
            host_to_secret_key("my-git.example.org"),
            "GIT_CREDENTIAL_MY_GIT_EXAMPLE_ORG"
        );
    }
}

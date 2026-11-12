//! The command-line surface: flags, and the commands that only read or toggle state rather than prove anything.
//!
//! Everything here talks to the running prover's own HTTP API on loopback when it can (`/mine`, `/job`), so `status` and
//! `mine on/off` describe and change the process that is actually working, not a guess from files on disk.

use std::time::Duration;

/// Parsed arguments: the command, then flags, then bare words.
///
/// Deliberately tiny. The prover has no subcommand nesting and never will, so `--key value`, `--key=value` and `--flag` are
/// the whole grammar; anything else is a bare word, kept in order for commands that take a path or an address.
pub struct Args {
    pub cmd: String,
    flags: Vec<(String, Option<String>)>,
    words: Vec<String>,
}

impl Args {
    pub fn parse(raw: &[String]) -> Self {
        let mut cmd = String::new();
        let mut flags = Vec::new();
        let mut words = Vec::new();
        let mut it = raw.iter().skip(1).peekable();
        while let Some(a) = it.next() {
            if let Some(rest) = a.strip_prefix("--") {
                let (k, v) = match rest.split_once('=') {
                    Some((k, v)) => (k.to_string(), Some(v.to_string())),
                    None => (rest.to_string(), None),
                };
                // A `--key value` pair only when the next word is not itself a flag and the key takes a value.
                let takes_value = matches!(k.as_str(), "miner" | "salt" | "port" | "api" | "dir" | "threads" | "every");
                let v = if v.is_some() {
                    v
                } else if takes_value {
                    match it.peek().and_then(|n| (!n.starts_with("--")).then(|| (*n).clone())) {
                        Some(n) => {
                            it.next();
                            Some(n)
                        }
                        None => None,
                    }
                } else {
                    None
                };
                flags.push((k, v));
            } else if cmd.is_empty() {
                cmd = a.clone();
            } else {
                words.push(a.clone());
            }
        }
        Args { cmd, flags, words }
    }

    pub fn flag(&self, k: &str) -> Option<&str> {
        self.flags.iter().find(|(n, _)| n == k).and_then(|(_, v)| v.as_deref())
    }

    pub fn has(&self, k: &str) -> bool {
        self.flags.iter().any(|(n, _)| n == k)
    }

    /// The first bare word, which for `prove` is the input path.
    pub fn word(&self, i: usize) -> Option<&str> {
        self.words.get(i).map(String::as_str)
    }

    /// Every bare word, in order, for commands that read a list (`mine on 0x...`).
    pub fn words(&self) -> Vec<String> {
        self.words.clone()
    }

    /// Refuse a flag that a command does not read, rather than ignoring it. A typo'd `--salt 0` that silently does nothing
    /// is worse than an error, because the next thing the user does is wonder why the CPU is still busy.
    pub fn allow(&self, known: &[&str]) -> Result<(), String> {
        for (k, _) in &self.flags {
            if !known.contains(&k.as_str()) {
                return Err(format!(
                    "unknown option --{k}{}",
                    if known.is_empty() {
                        String::new()
                    } else {
                        format!("; this command takes {}", known.iter().map(|f| format!("--{f}")).collect::<Vec<_>>().join(", "))
                    }
                ));
            }
        }
        Ok(())
    }
}

/// The loopback port the running prover listens on.
pub fn port() -> u16 {
    std::env::var("TACIT_PROVER_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(47815)
}

/// One request to the running prover, or None when nothing is listening.
fn ask(method: &str, path: &str, body: Option<serde_json::Value>) -> Option<(u16, serde_json::Value)> {
    let url = format!("http://127.0.0.1:{}{path}", port());
    let c = reqwest::blocking::Client::builder().timeout(Duration::from_secs(5)).build().ok()?;
    let r = match (method, body) {
        ("POST", Some(b)) => c.post(&url).json(&b).send().ok()?,
        _ => c.get(&url).send().ok()?,
    };
    let code = r.status().as_u16();
    Some((code, r.json().unwrap_or(serde_json::Value::Null)))
}

/// The pool's own view of the world, straight from its public endpoint.
fn pool_stats(api: &str) -> Option<serde_json::Value> {
    let c = reqwest::blocking::Client::builder().timeout(Duration::from_secs(8)).build().ok()?;
    c.get(format!("{api}/prover/pool")).send().ok()?.json().ok()
}

fn api() -> String {
    std::env::var("TACIT_POOL_API").unwrap_or_else(|_| "https://api.tacit.finance".into()).trim_end_matches('/').to_string()
}

fn num(v: &serde_json::Value, k: &str) -> u64 {
    v[k].as_u64().unwrap_or(0)
}

/// `status`: what this machine is doing and what the pool has, side by side, because the two together are the whole answer
/// to "am I earning" and either alone is misleading -- a prover that is up but idle and a pool with thousands of jobs for
/// other people both look like success from one angle.
pub fn status(json: bool, api_override: Option<&str>) -> Result<String, String> {
    let api = api_override.map(str::to_string).unwrap_or_else(api);
    let up = ask("GET", "/mine", None);
    let pool = pool_stats(&api);

    if json {
        return Ok(serde_json::json!({
            "running": up.is_some(),
            "prover": up.as_ref().map(|(_, v)| v.clone()),
            "pool": pool,
        })
        .to_string());
    }

    let mut out = String::new();
    match &up {
        Some((_, v)) => {
            let miner = v["miner"].as_str().unwrap_or("(none set)");
            let on = v["on"].as_bool().unwrap_or(false);
            out.push_str(&format!("prover   up on 127.0.0.1:{}\n", port()));
            out.push_str(&format!("miner    {miner}\n"));
            out.push_str(&format!("mining   {}\n", if on { "on" } else { "paused" }));
            out.push_str(&format!("proved   {} this run\n", num(v, "proved")));
            match v["job"].as_object() {
                Some(j) => {
                    out.push_str(&format!(
                        "job      {} — {}s{} \n",
                        j["id"].as_str().unwrap_or("?"),
                        j["secs"].as_f64().unwrap_or(0.0) as i64,
                        j["stage"].as_str().map(|s| format!(", {s}{}", j["eta"].as_f64().map(|e| format!(" ({}s left)", e as i64)).unwrap_or_default())).unwrap_or_default()
                    ));
                }
                None => out.push_str("job      none — waiting for the pool to lease one\n"),
            }
            // `rate` is hashes a second and `today` is an object with the day's shares and weight, not a count.
            let salt = &v["salt"];
            let rate = salt["rate"].as_f64().unwrap_or(0.0);
            let shares = salt["today"]["shares"].as_u64().unwrap_or(0);
            let weight = salt["today"]["weight"].as_u64().unwrap_or(0);
            out.push_str(&format!(
                "salt     {}{}{}\n",
                if salt["on"].as_bool() == Some(false) {
                    "off".to_string()
                } else if rate >= 1e6 {
                    format!("{:.1}M hashes/s", rate / 1e6)
                } else if rate >= 1e3 {
                    format!("{:.0}k hashes/s", rate / 1e3)
                } else {
                    format!("{rate:.0} hashes/s")
                },
                if shares > 0 { format!(", {shares} shares today") } else { String::new() },
                if weight > 0 { format!(" (weight {weight}, {} bits)", salt["minBits"].as_u64().unwrap_or(0)) } else { String::new() },
            ));
        }
        None => {
            out.push_str(&format!("prover   not running (nothing on 127.0.0.1:{})\n", port()));
        }
    }

    if let Some(p) = &pool {
        let done = num(p, "doneToday");
        let miners = num(p, "minersToday");
        out.push_str(&format!(
            "pool     {} queued, {} leased — {done} proofs today from {miners} miners\n",
            num(p, "queued"),
            num(p, "leased")
        ));
        let per = p["terms"]["pointsPerProof"].as_u64().unwrap_or(0);
        let lease = p["terms"]["leaseMs"].as_u64().unwrap_or(0) / 1000;
        out.push_str(&format!("terms    {per} points a proof, a lease is held {lease}s\n"));
        if let Some(top) = p["top"].as_array() {
            if let Some(first) = top.first() {
                out.push_str(&format!(
                    "leader   {} with {} proofs\n",
                    first["miner"].as_str().unwrap_or("?"),
                    num(first, "proofs")
                ));
            }
            if let Some(mine) = up.as_ref().and_then(|(_, v)| v["miner"].as_str()) {
                let rank = top.iter().position(|t| t["miner"].as_str().map(|m| m.eq_ignore_ascii_case(mine)).unwrap_or(false));
                match rank {
                    Some(i) => out.push_str(&format!("you      #{} on the board with {} proofs\n", i + 1, num(&top[i], "proofs"))),
                    None => out.push_str("you      not on the board yet\n"),
                }
            }
        }
    } else {
        out.push_str("pool     unreachable\n");
    }
    Ok(out)
}

/// `mine on|off|<address>`: turn the pool job loop on or off without restarting the prover, and optionally set the address
/// the points go to. This is the answer to "stop mining while I use the machine" without losing the GPU setup.
pub fn set_mining(words: &[String]) -> Result<String, String> {
    let (on, miner) = match words.first().map(String::as_str) {
        Some("off") | Some("stop") | Some("false") => (false, words.get(1).cloned()),
        Some("on") | Some("start") | Some("true") => (true, words.get(1).cloned()),
        Some(a) if a.starts_with("0x") => (true, Some(a.to_string())),
        None => return Err("usage: tacit-prover mine on|off [0x<address>]".into()),
        Some(o) => return Err(format!("mine takes on, off or an 0x address, not {o}")),
    };
    if let Some(m) = &miner {
        if !crate::pool::is_address(m) {
            return Err(format!("{m} is not a 0x address of 40 hex digits"));
        }
    }
    let mut body = serde_json::json!({ "on": on });
    if let Some(m) = miner.clone() {
        body["miner"] = m.into();
    }
    match ask("POST", "/mine", Some(body)) {
        Some((200, v)) => Ok(format!(
            "mining {}{}",
            if on { "on" } else { "paused" },
            v["miner"].as_str().map(|m| format!(" for {m}")).unwrap_or_default()
        )),
        Some((code, v)) => Err(format!("the prover refused it ({code}): {}", v["error"].as_str().unwrap_or("no reason given"))),
        None => Err(format!(
            "nothing is listening on 127.0.0.1:{}, so there is no running prover to change; start one with `tacit-prover serve`",
            port()
        )),
    }
}

/// `watch`: status, refreshed, until interrupted. Cheap enough (two requests a second) to leave open in a terminal.
pub fn watch(every: Duration, api_override: Option<&str>) -> Result<(), String> {
    loop {
        print!("\x1b[2J\x1b[H");
        match status(false, api_override) {
            Ok(s) => print!("{s}"),
            Err(e) => println!("{e}"),
        }
        println!("\n{}", now_string());
        std::thread::sleep(every);
    }
}

fn now_string() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // UTC clock without a date dependency: the prover only needs to know how fresh this is.
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02} UTC")
}

/// A one-line usage block, kept beside the parsing so it cannot drift.
pub const USAGE: &str = "\
tacit-prover — prove the pool's jobs, or your own

  serve                        run the prover and the local API (default)
  prove <stdin.bin>            prove one input file, write <stdin.bin>.proof.json
  status [--json] [--api URL]  what this machine and the pool are doing
  watch [--every SECS]         status, refreshed, until interrupted
  mine on|off [0x<address>]    start or pause pool mining without restarting
  check                        the GPU kernels against the CPU, and a proof
  install | uninstall          start at login, or stop doing that

options
  --miner 0x<address>          the address the TAC points go to
  --salt N                     cores for salt grinding while waiting (0 is off)
  --port N                     the local API port (default 47815)
  --api URL                    the pool API (default https://api.tacit.finance)
  --dir PATH                   where proofs, logs and timing live (default ~/.tacit-prover)
  --help                       this

environment
  TACIT_MINE, TACIT_SALT_THREADS, TACIT_PROVER_PORT, TACIT_POOL_API, TACIT_PROVER_DIR,
  SP1_GNARK_GPU=1 (the Groth16 multi-exponentiations on the GPU), TACIT_STAGE_TIMES=1
";

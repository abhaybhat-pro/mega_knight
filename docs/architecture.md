# Architecture

Synthetic data only. Not legal advice. Numbers quoted anywhere come from files under `docs/evidence/`.

```
generate (seed 42) -> shards (8 RandomForests, person_id hash -> shard) -> served ensemble (mean of shards)
                                   |                                          |
                          erase: retrain ONLY the affected shard      Aegis-Z Guard on incoming batches
                                   |                                (PSI drift, entropy, subgroup AUROC)
                 membership-inference audit before / after                     |
                                   |                                 hold-for-review (a person decides)
              signed receipt (HMAC-SHA256) + hash-chained JSONL log   |
                                   |                                  |
                         plain-language report (template; Granite is Planned) -> number validator
```

- **Data** (`src/data`): 20,000 training rows including 400 member canaries, 400 held-out canaries, 2,000 test rows, a drift stream.
- **Shards** (`src/model/shards.py`): shard k is a RandomForest (100 trees, `random_state` 42+k). A person id hashes to one shard, so erasing a person retrains one shard.
- **Audit** (`src/audit/mia.py`): a shadow-model attack on the canaries. It reports an attack AUC with a bootstrap interval, before and after erase. It is evidence from this attack, not a proof.
- **Erase** (`src/erase/queue.py`): validates the request, removes the rows, retrains the affected shard, audits, writes the receipt and the log entry.
- **Receipt and log** (`src/report`): the HMAC key comes only from `FORGETME_HMAC_KEY`. Receipts and logs hold counts and metrics, never patient features or labels.
- **Guard** (`src/guard/monitor.py`): the outcomes are `proceed`, `warn` and `hold-for-review`. There is no automatic refusal outcome. The hold list is persons in villages whose max feature PSI is at least 0.25 and it lives in memory only; the receipt carries only counts (`guard.drift_flags`, `guard.held_decisions`).
- **UI** (`app/`): Streamlit. plotly is optional and used in `app/` only; the pages fall back to native charts without it.
- **Facade** (`src/forgetme_api.py`): short names over the functions above. `make_data` returns `(df, {"member": [...], "heldout": [...]})`.

Target machine: a LinuxONE s390x VM (Planned, see `docs/vm_runbook.md`). All measurements so far are x86_64 rehearsals.

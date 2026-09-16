# oci-jhb-slurm-1 cluster reference

Saved 2026-09-10 from the Confluence page content supplied by the user.
This is a readable transcription of the operational information, not the original
Confluence layout XML. Empty template fields in the source are marked unspecified
or omitted; they must not be interpreted as zero limits.

## Hardware and access

- Oracle, Linux, GB300; MARS cluster, not MARS on Lepton.
- Source status: M1. Current status: <https://aihub.nvidia.com/clusters/oci-jhb-slurm-1>.
- Login: `oci-jhb-slurm-1-login-01.nvidia.com`, `oci-jhb-slurm-1-login-02.nvidia.com`.
- Data copying: `oci-jhb-slurm-1-dc-01.nvidia.com` through `-03.nvidia.com`.
- VS Code: `oci-jhb-slurm-1-vscode-01.nvidia.com`, `oci-jhb-slurm-1-vscode-02.nvidia.com`.
- Run VS Code only on the dedicated VS Code nodes, not login/data copier nodes.
- Access requires cluster onboarding and the welcome email. VPN Microsoft MFA/SSO:
  Austin, Helsinki, ngvpn01, ngvpn02, ngvpn07, ngvpn30, Wurselen; campus access:
  Endeavor and Voyager using Microsoft MFA/SSO. SSH multiplexing is recommended.
- Detailed GPU/CPU clocks, RAM, core counts, and node counts were blank in the source.

## Scheduling

Slurm 24.11 or higher; specify both partition and QoS. Whole nodes are required.
Use `--segment` when needed for topology placement.

| Partition | Purpose / maximum wall time |
| --- | --- |
| `batch` | Up to 4 hours |
| `batch_long` | Between 4 hours and 7 days |
| `cpu` | CPU only, up to 7 days (QoS can impose a shorter limit) |
| `cpu_datamover` | Inter-cluster transfers, no partition wall limit |

| QoS | Priority | Wall limit in source | Other limits / behavior |
| --- | --- | --- | --- |
| `normal` | 100 | Unspecified | Grace 20 min; preempts free/self; exemption 4h05m; within/requeue |
| `free` | 0 | Unspecified | Grace 10 min; exemption 30 min; requeue; usage factor 0 |
| `interactive` | 700 | Unspecified | Max 4 nodes/user; grace 20 min; exemption 4h05m; within/cancel |
| `short` | 200 | 2 hours | Max 64 nodes/user; preempts free; no preemption for jobs ≤2 hours |
| `cpu-normal` | 500 | 24 hours | No GPUs; usage factor 0 |
| `cpu-short` | 1000 | 4 hours | No GPUs |
| `cpu-long` | 500 | 7 days | No GPUs |
| `cpu-interactive` | 1200 | 24 hours | No GPUs |
| `cpu-dataprocessing` | 1100 | Unspecified | No GPUs |

All listed QoS entries carry `DenyOnLimit`. GPU MinTRES fields were blank in the
source. One introductory table spells `cpu_normal`; the detailed QoS table spells
`cpu-normal`. Query live associations for available names and actual limits.

## Communication stack

- Recommended NCCL: 2.30.7 for maximum CX8 line speed. Source reports a possible
  OOM tracked at <https://nvbugs/6521277>.
- Recommended HPCX: 2.26 for SPCX plugin and CX8 + SpectrumX performance.
- Recommended container: `nvcr.io/nvidia/pytorch:26.06-py3`, containing HPCX 2.26
  and NCCL 2.30.5 according to the supplied page.

The source explicitly says **do not override these cluster-managed parameters**:

| Variable | Value |
| --- | --- |
| `NCCL_IB_HCA` | `=rdma_vf_rail0,rdma_vf_rail1,rdma_vf_rail2,rdma_vf_rail3` |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET6` |
| `NCCL_IB_TIMEOUT` | `20` |
| `NCCL_SOCKET_IFNAME` | `eth0` |
| `NCCL_IB_ADAPTIVE_ROUTING` | `1` |
| `NCCL_DEBUG` | `WARN` |
| `NCCL_NET_PLUGIN` | `spcx` |

Tunable: `NCCL_P2P_NET_CHUNKSIZE` (source describes a default of 128 KB;
larger values trade more memory for potential throughput). `FI_LOG_LEVEL=Warn`
for initial diagnostics, `Info` for detail with a performance cost.

Fabric: multi-plane, two tiers, 1:1 T0:T1 oversubscription; 126 × 200G links
between T0 and GB300 racks, 64 × 400G links between T0 and T1. The source's
network diagram attachment was not included in the pasted content.

## Storage

| Path | Usage |
| --- | --- |
| `/raid/scratch` | 21 TB local job storage; cleared between jobs |
| `/home/<username>` | 10 GB; configuration/source, not high-performance job I/O |
| `/lustre/fsw/portfolios/<portfolio>/users/<username>` | Shared Lustre, user-owned, project members can read/write |
| `/lustre/fsw/portfolios/<portfolio>` | Shared project Lustre for high-performance I/O |

Lustre size/inode quotas were unspecified. Inspect using
`/cm/shared/apps/scripts/fs-quota-status`.

Stale-data policy: warning at 80% of block allowance (aging + stale data);
individual and PPP blocking thresholds are currently N/A in the supplied page.
PPP stale allowances depend on filesystem fullness and project quota.

## Support and references

- Cluster status: <https://aihub.nvidia.com/clusters/oci-jhb-slurm-1>
- Support: <https://nvidia.enterprise.slack.com/archives/C0BDCCLGF7F>, tag
  `@oci-jhb-slurm-1--support`; complete the Slackbot prompt for a Jira ticket.
- Self-service: <https://nvidia.glean.com/chat>, or `@AskNV` in support channels.
- MARS support request: <https://requests-navigator.atlassian.net/servicedesk/customer/portal/11/group/48/create/111>
- Storage increase: <https://requests-navigator.atlassian.net/servicedesk/customer/portal/11/group/48/create/218>
- Onboarding: <https://nvidia.atlassian.net/wiki/x/ao_IkQ>
- Login guidance: <https://nvidia.atlassian.net/wiki/x/FZOIkQ>
- Running jobs: <https://nvidia.atlassian.net/wiki/x/y5SIkQ>
- QoS: <https://nvidia.atlassian.net/wiki/x/M4KIkQ>
- QoS rationale: <https://nvidia.atlassian.net/wiki/x/nZmIkQ>
- Distributed training optimization: <https://nvidia.atlassian.net/wiki/x/ZpaIkQ>
- Best practices: <https://nvidia.atlassian.net/wiki/x/fo_IkQ>
- Data mover: <https://nvidia.atlassian.net/wiki/x/ozTKq>
- Stale data policy: <https://nvidia.atlassian.net/wiki/x/qZeIkQ>
- FAQs: <https://nvidia.atlassian.net/wiki/x/68jyuQ>
- Knowledge exchange sessions: <https://nvidia.atlassian.net/wiki/x/sJaIkQ>
- Benchmarks: <https://stormbreaker-dashboard.nvidia.com/training?tab=nemotron&time=90d&clusters=oci-jhb-slurm-1>

## Locally verified during setup

- Login environment: `oci-jhb-slurm-1-vscode-02`, `aarch64`.
- `sinfo`: GPU partitions expose 4 GPUs/node; `batch` 4h, `batch_long` 7d.
- User `apanferov` has accounts `coreai_psx_qad` and `coreai_psx_nextgen`.
  Use `coreai_psx_qad` for this project.
- Live association includes `cpu-normal`, `cpu-short`, `cpu-long`,
  `cpu-interactive`, `normal`, `short`, `interactive`, and `free`.
- `readlink -f /lustre` and `readlink -f /scratch` both return `/scratch`.

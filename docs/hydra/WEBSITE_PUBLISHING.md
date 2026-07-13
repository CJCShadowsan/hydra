# Publishing hydra-llm.cloud

Hydra's public website source lives in `website/src`. The Eleventy build emits
static output into `docs/`, and `.github/workflows/website-pages.yml` deploys a
GitHub Pages artifact from that generated output.

## Repository settings

1. Commit and push website source changes to `main`.
2. In `CJCShadowsan/hydra`, open **Settings -> Pages**.
3. Set **Build and deployment** to **GitHub Actions**.
4. Set **Custom domain** to `hydra-llm.cloud`.
5. Wait for the Pages workflow to deploy.
6. After DNS and certificate provisioning complete, enable **Enforce HTTPS**.

GitHub also exposes this through the Pages API:

```bash
gh api repos/CJCShadowsan/hydra/pages \
  -X POST \
  -f build_type=workflow

gh api repos/CJCShadowsan/hydra/pages \
  -X PUT \
  -f cname=hydra-llm.cloud \
  -f build_type=workflow
```

If the site already exists, the `POST` can return a conflict; run the `PUT`.

## Verify the domain

Before or immediately after adding the custom domain, verify `hydra-llm.cloud`
in GitHub account Pages settings. GitHub will show a TXT value. Add it in
Namecheap:

| Type | Host | Value |
|---|---|---|
| `TXT` | `_github-pages-challenge-CJCShadowsan` | value shown by GitHub |

Keep this TXT record. It prevents other GitHub users from claiming the domain
or its immediate subdomains if the Pages site is ever disabled.

## Namecheap DNS

In Namecheap, open **Domain List -> hydra-llm.cloud -> Manage -> Advanced DNS**.
Remove parking/redirect defaults that conflict with these records, then add:

| Type | Host | Value |
|---|---|---|
| `A` | `@` | `185.199.108.153` |
| `A` | `@` | `185.199.109.153` |
| `A` | `@` | `185.199.110.153` |
| `A` | `@` | `185.199.111.153` |
| `AAAA` | `@` | `2606:50c0:8000::153` |
| `AAAA` | `@` | `2606:50c0:8001::153` |
| `AAAA` | `@` | `2606:50c0:8002::153` |
| `AAAA` | `@` | `2606:50c0:8003::153` |
| `CNAME` | `www` | `CJCShadowsan.github.io` |

Use Namecheap's automatic TTL unless you need faster later changes. Do not add
wildcard records such as `*.hydra-llm.cloud`.

## Check propagation

DNS can take up to 24 hours to propagate.

```bash
dig hydra-llm.cloud +noall +answer -t A
dig hydra-llm.cloud +noall +answer -t AAAA
dig www.hydra-llm.cloud +nostats +nocomments +nocmd
```

Expected:

- `hydra-llm.cloud` resolves to GitHub Pages `A` and optional `AAAA` records.
- `www.hydra-llm.cloud` is a CNAME to `CJCShadowsan.github.io`.
- GitHub Pages shows the custom domain as DNS-valid.
- **Enforce HTTPS** becomes available after certificate provisioning.

## References

- GitHub Pages custom domains:
  <https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site>
- GitHub Pages domain verification:
  <https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/verifying-your-custom-domain-for-github-pages>
- GitHub Pages custom workflows:
  <https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>

const site = {
  title: 'Hydra',
  description: 'Hydra is a network-aware fork of Mesh LLM for low-latency distributed inference.',
  url: 'https://hydra-llm.cloud',
  githubUrl: 'https://github.com/CJCShadowsan/hydra',
  githubRepo: 'CJCShadowsan/hydra',
  githubStarsFallback: '1.1k',
  githubReleaseFallback: 'source',
};

const fetchLatestReleaseTag = async (repo) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`https://api.github.com/repos/${repo}/releases/latest`, {
      headers: {
        Accept: 'application/vnd.github+json',
        'User-Agent': 'hydra-website',
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    if (!response.ok) {
      console.warn(`GitHub API returned ${response.status}; using fallback release version`);
      return null;
    }

    const release = await response.json();
    const tagName = typeof release?.tag_name === 'string' ? release.tag_name.trim() : '';
    return tagName || null;
  } catch (err) {
    console.warn('Failed to fetch GitHub release, falling back to hardcoded version:', err);
    return null;
  }
};

export default async function () {
  const githubReleaseFallback = await fetchLatestReleaseTag(site.githubRepo) ?? site.githubReleaseFallback;

  return {
    ...site,
    githubReleaseFallback,
  };
}

import { defineConfig } from 'vitepress'

// VitePress configuration for the hybridmodels documentation site.
//
// Layout follows the gsax docs: a flat sidebar per section (Guide,
// Examples, API), local search, and math rendering enabled so the
// crystallisation example's ODE expressions and the moments equations
// render cleanly.
export default defineConfig({
  title: 'hybridmodels',
  description: 'Hybrid models in JAX — trainable predictors composed with user-written ODE dynamics.',
  base: '/jax-hybridmodels/',

  markdown: {
    math: true,
  },

  // The repo's `docs/adr/` and `docs/agents/` directories carry internal
  // architectural decisions and agent prompts — relative links inside them
  // (`../../SPEC`, `../adr/index`) only resolve from the repo root, not
  // from the docs site. Exclude them from the VitePress scan.
  srcExclude: ['adr/**', 'agents/**'],

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Examples', link: '/examples/crystallisation-notebook' },
      { text: 'API', link: '/api/' },
    ],

    sidebar: {
      '/guide/': [
        { text: 'Getting Started', link: '/guide/getting-started' },
        { text: 'Concepts', link: '/guide/concepts' },
        { text: 'Training', link: '/guide/training' },
        { text: 'Recommendations', link: '/guide/recommendations' },
      ],
      // The two embedded marimo notebooks are the canonical example docs.
      // The earlier hybrid-MLP and mechanistic crystallisation walk-throughs
      // and the harmonic-oscillator page are intentionally not listed here —
      // their content has been folded into the combined crystallisation
      // notebook. The pages still exist on disk; they're just unlinked.
      '/examples/': [
        { text: 'Crystallisation', link: '/examples/crystallisation-notebook' },
        { text: 'Batch reactor', link: '/examples/batch-reactor' },
        { text: 'Neural ODE and hybrid', link: '/examples/neural-ode' },
      ],
      '/api/': [
        { text: 'Overview', link: '/api/' },
        { text: 'Data', link: '/api/data' },
        { text: 'Predictors', link: '/api/predictors' },
        { text: 'Penalties', link: '/api/penalties' },
        { text: 'Transforms', link: '/api/transforms' },
        { text: 'Solver', link: '/api/solver' },
        { text: 'Training', link: '/api/training' },
        { text: 'Losses', link: '/api/losses' },
        { text: 'Trainable Masks', link: '/api/trainable' },
        { text: 'Prediction', link: '/api/prediction' },
        { text: 'Serialise', link: '/api/serialise' },
        { text: 'UI', link: '/api/ui' },
        { text: 'RNG', link: '/api/rng' },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/DanielePessina/jax-hybridmodels' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
    },
  },
})

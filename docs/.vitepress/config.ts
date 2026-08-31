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
      { text: 'Examples', link: '/examples/custom-loop' },
      { text: 'API', link: '/api/' },
    ],

    sidebar: {
      '/guide/': [
        { text: 'Getting Started', link: '/guide/getting-started' },
        { text: 'Concepts', link: '/guide/concepts' },
        { text: 'Model interface', link: '/guide/model-interface' },
        { text: 'Data and buckets', link: '/guide/data' },
        { text: 'Predictors and bounds', link: '/guide/predictors' },
        { text: 'Training', link: '/guide/training' },
        { text: 'Custom Predictors', link: '/guide/custom-predictors' },
        { text: 'Extending', link: '/guide/extending' },
        { text: 'Ensembles', link: '/guide/ensembles' },
        { text: 'Saving and loading', link: '/guide/serialization' },
        { text: 'Troubleshooting', link: '/guide/troubleshooting' },
        { text: 'Recommendations', link: '/guide/recommendations' },
      ],
      // Examples are plain scripts single-sourced into the docs. The
      // earlier marimo notebook pages were converted to scripts and the
      // notebook-specific pages removed.
      '/examples/': [
        { text: 'Custom training loop', link: '/examples/custom-loop' },
        { text: 'Crystallisation', link: '/examples/crystallisation' },
        { text: 'Hybrid ODE', link: '/examples/hybrid-ode' },
        { text: 'Batch reactor', link: '/examples/batch-reactor' },
        { text: 'Batch reactor RL', link: '/examples/batch-reactor-rl' },
        { text: 'SBML hybrid kinetics', link: '/examples/sbml-hybrid' },
        { text: 'Custom predictor', link: '/examples/custom-predictor' },
        { text: 'Neural polynomial kinetics', link: '/examples/supersaturation-poly' },
      ],
      '/api/': [
        { text: 'Overview', link: '/api/' },
        { text: 'Data', link: '/api/data' },
        { text: 'Predictors', link: '/api/predictors' },
        { text: 'Penalties', link: '/api/penalties' },
        { text: 'Profiles', link: '/api/profiles' },
        { text: 'Schedules', link: '/api/schedules' },
        { text: 'Transforms', link: '/api/transforms' },
        { text: 'Solver', link: '/api/solver' },
        { text: 'Training', link: '/api/training' },
        { text: 'Training Kernels', link: '/api/kernels' },
        { text: 'Losses', link: '/api/losses' },
        { text: 'Metrics', link: '/api/metrics' },
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

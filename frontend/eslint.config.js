import nextPlugin from 'eslint-config-next';

export default [
  {
    ignores: ['.next', 'node_modules', 'out', 'build', '*.config.js'],
  },
  {
    files: ['**/*.{js,jsx,ts,tsx}'],
    plugins: nextPlugin.plugins || {},
    rules: {
      ...nextPlugin.rules,
    },
  },
];

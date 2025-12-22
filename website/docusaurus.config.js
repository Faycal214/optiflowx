// website/docusaurus.config.js
module.exports = {
  title: 'OptiFlowX',
  tagline: 'Research-grade automatic ML pipeline optimizer',
  url: 'https://your-domain.com',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'static/img/opti-mark.svg',
  organizationName: 'Faycal214',
  projectName: 'optiflowx',
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: '../docs', // use existing docs/ in repo root
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/Faycal214/optiflowx/edit/main/docs/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'OptiFlowX',
      logo: {
        alt: 'OptiFlowX mark',
        src: 'static/img/opti-mark.svg',
      },
      items: [
        {
          href: 'https://github.com/Faycal214/optiflowx',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    prism: {
      theme: require('prism-react-renderer/themes/vsDark'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
      additionalLanguages: ['python'],
    },
  },
  clientModules: [require.resolve('./src/clientModules.js')],
};

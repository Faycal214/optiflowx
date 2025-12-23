// website/docusaurus.config.js
module.exports = {
  title: 'OptiFlowX',
  tagline: 'Research-grade automatic ML pipeline optimizer',
  url: 'https://Faycal214.github.io',
  baseUrl: '/optiflowx/',
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
      title: 'OptiFlowX',         // keep simple text title or remove title if preferred
      // remove or comment out any `logo` entry so no image mark is shown
      items: [
        {to: '/', label: 'Docs', position: 'left'},
        {href: 'https://github.com/Faycal214/optiflowx', label: 'GitHub', position: 'right'}
      ],
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: true,
      respectPrefersColorScheme: false
    },
    prism: {
      // use a dark base; we will override tokens via CSS for richer colors
      theme: require('prism-react-renderer/themes/dracula'),
      darkTheme: require('prism-react-renderer/themes/dracula')
    },
    // ensure no clientModules reference pointing to a theme toggle component
  },
};

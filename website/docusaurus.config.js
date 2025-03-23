// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {{onBrokenLinks: string, organizationName: string, plugins: string[], title: string, url: string, onBrokenMarkdownLinks: string, i18n: {defaultLocale: string, locales: string[]}, trailingSlash: boolean, baseUrl: string, presets: [string,Options][], githubHost: string, tagline: string, themeConfig: ThemeConfig & UserThemeConfig & AlgoliaThemeConfig, projectName: string}} */
const config = {
  title: 'AI on EKS',
  tagline: 'Supercharge your AI/ML Journey with Amazon EKS',
  url: 'https://awslabs.github.io',
  baseUrl: '/ai-on-eks/',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/header-icon.png',

  organizationName: 'awslabs',
  projectName: 'ai-on-eks',
  githubHost: 'github.com',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/awslabs/ai-on-eks/blob/main/website/',
        },
        theme: {
          customCss: [
            require.resolve('./src/css/custom.css'),
            require.resolve('./src/css/fonts.css'),
          ],
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        // title: 'DoEKS',
        logo: {
          alt: 'DoEKS Logo',
          src: 'img/header-icon.png',
        },
        items: [
          { type: 'doc', docId: 'ai/index', position: 'left', label: 'AI' },
          { type: 'doc', docId: 'blueprints/ai-ml/index', position: 'left', label: 'Blueprints' },
          { type: 'doc', docId: 'bestpractices/intro', position: 'left', label: 'Best Practices' },
          { type: 'doc', docId: 'resources/intro', position: 'left', label: 'Resources' },
          { href: 'https://github.com/awslabs/ai-on-eks', label: 'GitHub', position: 'right' },
        ],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        }
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Get Involved',
            items: [{ label: 'Github', href: 'https://github.com/awslabs/ai-on-eks' }],
          },
        ],
        copyright: `Built with ❤️ at AWS  <br/> © ${new Date().getFullYear()} Amazon.com, Inc. or its affiliates. All Rights Reserved`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),

    plugins: [require.resolve('docusaurus-lunr-search')],
};

module.exports = config;

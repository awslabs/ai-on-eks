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
    onBrokenLinks: 'warn',
    onBrokenMarkdownLinks: 'warn',
    favicon: 'img/header-icon.png',

    organizationName: 'awslabs',
    projectName: 'ai-on-eks',
    githubHost: 'github.com',

    i18n: {
        defaultLocale: 'en',
        locales: ['en', 'ko'],
        localeConfigs: {
            en: {
                label: 'English',
                htmlLang: 'en-US',
            },
            ko: {
                label: '한국어',
                htmlLang: 'ko-KR',
            },
        },
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

    themes: ['@docusaurus/theme-mermaid'],

    markdown: {
        mermaid: true,
    },

    themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            announcementBar: {
                id: 'genai-workshop-banner',
                content:
                    'GenAI on EKS workshop series! <a target="_blank" rel="noopener noreferrer" href="https://aws-experience.com/emea/smb/events/series/get-hands-on-with-amazon-eks?trk=9be4af2e-2339-40ae-b5e9-57b6a7704c36&sc_channel=el" style="color: #ffffff; text-decoration: underline; font-weight: bold; margin-left: 10px;">Register now →</a>',
                backgroundColor: '#667eea',
                textColor: '#ffffff',
                isCloseable: true,
            },
            mermaid: {
                theme: {light: 'neutral', dark: 'forest'},
                options: {
                    maxTextSize: 50000,
                },
            },
            navbar: {
                // title: 'AIoEKS',
                logo: {
                    alt: 'AIoEKS Logo',
                    src: 'img/header-icon.png',
                },
                items: [
                    {type: 'doc', docId: 'infra/index', position: 'left', label: 'Infrastructure'},
                    {type: 'doc', docId: 'blueprints/index', position: 'left', label: 'Blueprints'},
                    {type: 'doc', docId: 'guidance/index', position: 'left', label: 'Guidance'},
                    {href: 'https://github.com/awslabs/ai-on-eks', label: 'GitHub', position: 'right'},
                    {type: 'localeDropdown', position: 'right'},
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
                        items: [{label: 'Github', href: 'https://github.com/awslabs/ai-on-eks'}],
                    },
                ],
                copyright: `Built with ❤️ at AWS  <br/> © ${new Date().getFullYear()} Amazon.com, Inc. or its affiliates. All Rights Reserved`,
            },

            prism: {
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
                additionalLanguages: ['bash', 'yaml', 'hcl', 'json', 'python', 'javascript', 'typescript', 'jsx', 'tsx'],
            },
        }),

    plugins: [require.resolve('docusaurus-lunr-search'),
        ['@docusaurus/plugin-client-redirects', {
            createRedirects(existingPath) {
                if (existingPath.includes('/docs/guidance')) {
                    return [
                        existingPath.replace('/docs/guidance', '/docs/resources'),
                    ];
                }
                if (existingPath.includes('/docs/infra')) {
                    return [
                        existingPath.replace('/docs/infra', '/docs/infra/ai-ml'),
                    ];
                }
                return undefined; // Return a falsy value: no redirect created
            },
        }]],
};

module.exports = config;

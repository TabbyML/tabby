// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
export default {
  title: 'Tabby',
  tagline: 'Opensource, self-hosted AI coding assistant',
  favicon: 'img/favicon.ico',
  trailingSlash: true,

  // Set the production url of your site here
  url: 'https://tabby.tabbyml.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'TabbyML', // Usually your GitHub org/user name.
  projectName: 'tabby', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  headTags: [
    {
      tagName: 'link',
      attributes: {
        href: 'https://fonts.googleapis.com/css?family=Azeret Mono',
        rel: 'stylesheet'
      }
    }
  ],

  presets: [
    [
      'docusaurus-preset-openapi',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/TabbyML/tabby/edit/main/website',
          admonitions: {
            keywords: ['note', 'tip', 'info', 'caution', 'warning', 'danger', 'subscription'],
          },
        },
        api: {
          path: "static/openapi.json",
          routeBasePath: "/api"
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/TabbyML/tabby/edit/main/website',
          blogSidebarCount: 20,
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/tabby-social-card.png',
      docs: {
        sidebar: {
          autoCollapseCategories: true
        },
      },
      navbar: {
        logo: {
          alt: 'Tabby',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          { to: '/api', label: 'API', position: 'left' },
          {
            href: 'https://github.com/TabbyML/tabby',
            label: 'GitHub',
            position: 'right',
            className: "header-github-link",

          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Links',
            items: [
              {
                label: 'Docs',
                to: '/docs/welcome',
              },
              {
                label: 'API',
                to: '/api',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Slack',
                href: 'https://slack.tabbyml.com',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/TabbyML/tabby',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: "Status",
                href: "https://uptime.tabbyml.com"
              },
              {
                label: "Media Kit",
                href: "https://www.figma.com/community/file/1299817332961215434/tabby-mediakit"
              },
              {
                html: `<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=96661b6c-b6b6-4340-9ffb-dcc46d7b970a" />`
              }
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} TabbyML, Inc.`,
      },
      prism: {
        theme: prismThemes.palenight,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['toml', 'rust', 'scheme'],
      },
      colorMode: {
        defaultMode: "light",
        respectPrefersColorScheme: false,
        disableSwitch: true
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 5,
      },
    }),

  plugins: [
    async function tailwind(context, options) {
      return {
        name: "docusaurus-tailwindcss",
        configurePostCss(postcssOptions) {
          // Appends TailwindCSS and AutoPrefixer.
          postcssOptions.plugins.push(require("tailwindcss"));
          postcssOptions.plugins.push(require("autoprefixer"));
          return postcssOptions;
        },
      };
    },
    [
      "posthog-docusaurus",
      {
        apiKey: "phc_aBzNGHzlOy2C8n1BBDtH7d4qQsIw9d8T0unVlnKfdxB",
        appUrl: "https://app.posthog.com",
        enableInDevelopment: false,
      },
    ],
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {
            to: '/blog/2024/02/05/create-tabby-extension-with-language-server-protocol',
            from: '/blog/running-tabby-as-a-language-server'
          },
          {
            to: '/docs/quick-start/installation/docker',
            from: '/docs/self-hosting/docker'
          },
          {
            to: '/docs/extensions/installation/vscode',
            from: '/docs/extensions/vscode'
          },
          {
            to: '/docs/administration/context',
            from: '/docs/configuration'
          },
          {
            to: '/docs/welcome',
            from: '/docs/getting-started'
          },
          {
            to: '/docs/extensions/installation/vscode',
            from: '/docs/extensions'
          },
          {
            to: '/docs/extensions/installation/vscode',
            from: '/docs/extensions/installation'
          },
          {
            to: '/docs/quick-start/installation/docker',
            from: '/docs/installation'
          },
          {
            to: '/docs/administration/upgrade',
            from: '/docs/administration'
          },
          {
            to: '/docs/welcome',
            from: '/docs',
          },
          {
            to: '/docs/quick-start/installation/docker',
            from: '/docs/quick-start/installation'
          },
          {
            to: '/docs/references/programming-languages',
            from: '/docs/programming-languages'
          },
        ],
        createRedirects(existingPath) {
          // Create redirection from /docs/installation/* to /docs/quick-start/installation/*
          if (existingPath.startsWith("/docs/quick-start/installation")) {
            return [
              existingPath.replace("/docs/quick-start/installation", "/docs/installation"),
            ]
          }

          // Create redirection from /docs/quick-start/installation/* to /docs/references/cloud-deployment/*
          if (existingPath.startsWith("/docs/references/cloud-deployment/")) {
            return [
              existingPath.replace("/docs/references/cloud-deployment/", "/docs/quick-start/installation/"),
              existingPath.replace("/docs/references/cloud-deployment/", "/docs/installation/"),
            ]
          }
        }
      },
    ],
  ],

  scripts: [
    {
      src: "https://tally.so/widgets/embed.js",
      async: true
    }
  ]
};

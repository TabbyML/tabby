// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/palenight');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Tabby',
  tagline: 'Opensource, self-hosted AI coding assistant',
  favicon: 'img/favicon.ico',

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
        },
        api: {
          path: "static/openapi.json",
          routeBasePath: "/api"
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/TabbyML/tabby/edit/main/website',
          blogSidebarCount: 10,
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
          {to: '/playground', label: 'Playground', position: 'left'},
          {to: '/blog', label: 'Blog', position: 'left'},
          {to: '/api', label: 'API', position: 'left'},
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
                to: '/docs/getting-started',
              },
              {
                label: 'Playground',
                to: '/playground',
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
              }
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} TabbyML, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['toml', 'rust', 'scheme'],
      },
      colorMode: {
        defaultMode: "light",
        respectPrefersColorScheme: false,
        disableSwitch: true
      }
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
  ],
};

module.exports = config;

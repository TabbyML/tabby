// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/palenight');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Tabby',
  tagline: 'Opensource, self-hosted AI coding assistant',
  // FIXME(meng): favicon
  // favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://tabbyml.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/tabby',

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
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
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
      // FIXME(meng): set social card.
      // image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Tabby',
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
          {to: '/api', label: 'API', position: 'left'},
          // FIXME(meng): enable blog.
          // {to: '/blog', label: 'Blog', position: 'left'},
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
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/tabby',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/mzh1024',
              },
            ],
          },
          {
            title: 'More',
            items: [
              /*
              {
                label: 'Blog',
                to: '/blog',
              },
              */
              {
                label: 'GitHub',
                href: 'https://github.com/TabbyML/tabby',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} TabbyML, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
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

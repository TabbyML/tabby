import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';
import Head from '@docusaurus/Head';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <Head>
        <title>{siteConfig.title} - {siteConfig.tagline}</title>
      </Head>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <div className="flex flex-col md:flex-row justify-center items-center gap-4">
          <Link
            rel="noopener"
            className="button button--secondary button--lg"
            to="#tally-open=mZJ10o&tally-layout=modal&tally-width=720&tally-hide-title=0&tally-emoji-text=👋&tally-emoji-animation=wave">
            View Live Demo 🚀
          </Link>
          <Link
            className="button button--ghost button--lg flex flex-col items-center hover:opacity-80 font-medium"
            to="/docs/quick-start/installation/docker">
            Tutorial - 5min ⏱️
            <div className='w-full h-[2px] bg-green-200 rounded-sm'></div>
          </Link>
        </div>

        <img className="mt-5" src="img/demo.gif" />
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      description="Tabby is a self-hosted AI coding assistant, offering an open-source and on-premises alternative to GitHub Copilot">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}

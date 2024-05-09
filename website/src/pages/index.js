import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <div className="flex flex-col md:flex-row justify-center items-center gap-4">
          <Link
            rel="noopener"
            className="button button--secondary button--lg"
            to="https://links.tabbyml.com/live-demo">
            View Live Demo 🚀
          </Link>
          <Link
            className="button button--ghost button--lg flex flex-col items-center hover:opacity-80 font-medium"
            to="/docs/getting-started">
            Tutorial - 5min ⏱️
            <div className='w-full h-[2px] bg-green-200 rounded-sm'></div>
          </Link>
        </div>

        <img className="m-5" src="img/demo.gif" />
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}

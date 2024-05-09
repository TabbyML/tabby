import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <div className="flex justify-center items-center gap-4 max-w-md mx-auto">
          <Link
            className="button button--secondary button--md flex-1"
            to="https://links.tabbyml.com/live-demo">
              View Live Demo 🚀
          </Link>
          <Link
            className="button button--primary border-neutral-800 bg-neutral-800 flex-1 hover:opacity-[0.85]"
            to="/docs/getting-started">
            Tutorial - 5min ⏱️
          </Link>
        </div>

        <img className="m-5" src="img/demo.gif" />
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
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

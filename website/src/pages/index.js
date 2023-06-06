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
        {false && <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Tutorial - 5min ⏱️
          </Link>
        </div>}
        <div className="flex justify-center gap-1">
          <ExternalLink href="https://github.com/TabbyML/tabby" imgUrl="https://img.shields.io/github/stars/TabbyML/tabby?style=social" />
          <ExternalLink href="https://hub.docker.com/r/tabbyml/tabby" imgUrl="https://img.shields.io/docker/pulls/tabbyml/tabby?style=social" />
        </div>
        <img className="m-5" src="img/demo.gif" />
      </div>
    </header>
  );
}

function ExternalLink({href, imgUrl}) {
  return <a target="_blanks" href={href}>
    <img src={imgUrl} />
  </a>
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

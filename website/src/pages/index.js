import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import VideoGrid from '@site/src/components/VideoGrid/VideoGrid';
import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();

  return (
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
          <video playsInline autoPlay muted loop webkit-playsinline  defaultMuted id="heroVideo">
              <source src="vid/background.mp4" type="video/mp4"/>
          </video>
              <p
                  className='hero__subtitle'
                  style={{
                      position: "absolute",
                      fontSize: 18,
                      fontSmooth: 'auto',
                      marginBottom: 0,
                  }}>
                  {siteConfig.tagline}
              </p>
      </header>
);
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <div className="container">
          <HomepageFeatures />
        </div>
      </main>
    </Layout>
  );
}

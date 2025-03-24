import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import VideoGrid from '@site/src/components/VideoGrid/VideoGrid';
import styles from './index.module.css';
import Head from '@docusaurus/Head';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
    const OGMeta = () => (
        <Head>
            <meta name="og:image" content="img/aioeks-logo-green.png" />
        </Head>
    );
  return (
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
          {OGMeta()}
          <video playsInline autoPlay muted loop webkit-playsinline defaultMuted id="heroVideo">
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
              <div className={styles.buttons}>
                  <Link
                      className={clsx("button button--lg", styles.buttonSpinUp)}
                      to="/docs/ai">
                      Let's Spin Up
                  </Link>
                  <Link
                      className={clsx("button button--lg", styles.buttonGenAI)}
                      to="https://awslabs.github.io/data-on-eks">
                      Data on EKS
                  </Link>
              </div>
          </p>
      </header>
  );
}

export default function Home() {
    const {siteConfig} = useDocusaurusContext();
    return (
        <Layout
            title={`AI on EKS (AIoEKS)`}
            description="Tested AI/ML on Amazon Elastic Kubernetes Service">
        <HomepageHeader/>
            <main>
                <div className="container">
                    <HomepageFeatures/>
                </div>
            </main>
        </Layout>
    );
}

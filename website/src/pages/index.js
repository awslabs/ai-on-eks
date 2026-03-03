import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Translate, {translate} from '@docusaurus/Translate';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import VideoGrid from '@site/src/components/VideoGrid/VideoGrid';
import styles from './index.module.css';
import Head from '@docusaurus/Head';

function HomepageHeader() {
    const { siteConfig, i18n } = useDocusaurusContext();
    const doeksUrl = i18n.currentLocale === 'ko'
        ? 'https://awslabs.github.io/data-on-eks/'
        : 'https://awslabs.github.io/data-on-eks/';
    const OGMeta = () => (
        <Head>
            <meta name="og:image" content="https://awslabs.github.io/ai-on-eks/img/aioeks-logo-green.png" />
        </Head>
    );
    return (
        <header className={clsx('hero', styles.heroBanner)}>
            {OGMeta()}
            <div className={styles.heroContainer}>
                {/* Main Logo Section */}
                <div className={styles.logoSection}>
                    <img
                        src="img/aioeks-logo-green.png"
                        alt="AI on EKS"
                        className={styles.logoImage}
                    />
                </div>

                {/* Hero Content */}
                <div className={styles.heroContent}>
                    <p className={styles.heroSubtitle}>
                        <Translate id="homepage.tagline" description="The site tagline">
                            Supercharge your AI/ML Journey with Amazon EKS
                        </Translate>
                    </p>
                    <p className={styles.heroDescription}>
                        <Translate id="homepage.description.line1" description="Hero description line 1">
                            The comprehensive set of tools for running AI workloads on Amazon EKS.
                        </Translate>
                        <br />
                        <Translate id="homepage.description.line2" description="Hero description line 2">
                            Build, deploy, and scale your AI infrastructure with confidence.
                        </Translate>
                    </p>
                </div>

                {/* CTA Buttons */}
                <div className={styles.ctaSection}>
                    <Link
                        className={clsx(styles.primaryButton)}
                        to="/docs/blueprints/">
                        <span><Translate id="homepage.cta.getStarted">Get Started</Translate></span>
                        <svg className={styles.buttonIcon} width="20" height="20" viewBox="0 0 20 20" fill="none">
                            <path d="M10.75 8.75L14.25 12.25L10.75 15.75" stroke="currentColor" strokeWidth="1.5"
                                strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M19.25 12.25H5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
                                strokeLinejoin="round" />
                        </svg>
                    </Link>
                    <Link
                        className={clsx(styles.secondaryButton)}
                        to={doeksUrl}>
                        <Translate id="homepage.cta.exploreDoEKS">Explore Data on EKS</Translate>
                    </Link>
                </div>
                <script async src="https://js.storylane.io/js/v2/storylane.js"></script>
                <div className="sl-embed"
                    style={{
                        "position": "relative",
                        "paddingBottom": "calc(40.25% + 25px)",
                        "width": "100%",
                        "maxWidth": "1280px",
                        "transform": "scale(1)",
                        "margin": "0 auto"
                    }}
                >
                    <iframe loading="lazy" className="sl-demo"
                        src="https://app.storylane.io/demo/uybpeyefbrgn?embed=inline" name="sl-embed"
                        allow="fullscreen"
                        allowFullScreen
                        style={{
                            "position": "absolute",
                            "top": "0",
                            "left": "0",
                            "width": "100%!important",
                            "height": "100%!important",
                            "border": "1px solid rgba(63,95,172,0.35)",
                            "boxShadow": "0px 0px 18px rgba(26, 19, 72, 0.15)",
                            "borderRadius": "10px",
                            "boxSizing": "border-box",
                            "height": "100%",
                            "width": "100%"
                        }}
                    >
                    </iframe>
                </div>
            </div>

            {/* Background Elements */}
            <div className={styles.backgroundElements}>
                <div className={styles.bgCircle1}></div>
                <div className={styles.bgCircle2}></div>
                <div className={styles.bgCircle3}></div>
            </div>
        </header>
    );
}

function AIOnEKSHeader() {
    return (
        <div className={styles.aiOnEKSHeader}>
        </div>
    );
}

export default function Home() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout
            title={translate({id: 'homepage.title', message: 'AI on EKS (AIoEKS)'})}
            description={translate({id: 'homepage.metaDescription', message: 'Tested AI/ML on Amazon Elastic Kubernetes Service'})}>
            <HomepageHeader />
            <AIOnEKSHeader />
            <main>
                <div className="container">
                    <HomepageFeatures />
                </div>
            </main>
        </Layout>
    );
}

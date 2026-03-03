import React from 'react';
import clsx from 'clsx';
import Translate, {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

const FeatureList = [
    {
        title: translate({id: 'features.infrastructure.title', message: 'Infrastructure'}),
        Svg: require('@site/static/img/infra.svg').default,
        link: "/ai-on-eks/docs/infra",
        description: (
            <Translate id="features.infrastructure.description">
                Validated infrastructure for the latest generation of Artificial Intelligence workloads on EKS.
            </Translate>
        ),
    },
    {
        title: translate({id: 'features.blueprints.title', message: 'Blueprints'}),
        Svg: require('@site/static/img/blueprints.svg').default,
        link: "/ai-on-eks/docs/blueprints",
        description: (
            <Translate id="features.blueprints.description">
                Tested deployments to jumpstart and enable AI and ML workloads on EKS
            </Translate>
        ),
    },
    {
        title: translate({id: 'features.guidance.title', message: 'Guidance'}),
        Svg: require('@site/static/img/guidance.svg').default,
        link: "/ai-on-eks/docs/guidance",
        description: (
            <Translate id="features.guidance.description">
                Proven experience scaling AI and ML on EKS
            </Translate>
        ),
    },
];

function Feature({Svg, title, description, link}) {
    return (

        <div className={clsx('col col--4')}>
            <a href={link} style={{textDecoration: 'none'}}>
                <div className="text--center">
                    <Svg className={styles.featureSvg} style={{width: '40%'}} role="img"/>
                </div>
                <div className="text--center padding-horiz--md">
                    <h2><b>{title}</b></h2>
                    <p>{description}</p>
                </div>
            </a>
        </div>

    );
}

export default function HomepageFeatures() {
    return (
        <section className={styles.features}>
            <div className="container">
                <div className="row">
                    {FeatureList.map((props, idx) => (
                        <Feature key={idx} {...props} />
                    ))}
                </div>
            </div>
        </section>
    );
}

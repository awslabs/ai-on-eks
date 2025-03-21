import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
    {
        title: 'Infrastructure',
        Svg: require('@site/static/img/infra.svg').default,
        description: (
            <div>
                Validated infrastructure for the latest generation of Artificial Intelligence workloads on EKS.
            </div>
        ),
    },
  {
    title: 'Generative AI',
    Svg: require('@site/static/img/green-ai.svg').default,
    description: (
      <>
        Explore cutting-edge blueprints and patterns for deploying Generative AI models on EKS.
      </>
    ),
  },
  {
    title: 'ML',
    Svg: require('@site/static/img/ml.svg').default,
    description: (
        <>
            Traditional ML Workloads powered by EKS<br/>
        </>
    ),
  },
  // {
  //       title: 'Streaming Data Platforms',
  //       Svg: require('@site/static/img/green-stream.svg').default,
  //       description: (
  //           <>
  //               Building High-Scalability Streaming Data Platforms with Kafka, Flink, Spark Streaming, and More
  //           </>
  //       ),
  // },
  // {
  //       title: 'Schedulers',
  //       Svg: require('@site/static/img/green-schd.svg').default,
  //       description: (
  //           <>
  //               Optimizing Job Scheduling on EKS with Apache Airflow, Amazon MWAA, Argo Workflow, and More
  //           </>
  //       ),
  // },
  // {
  //       title: 'Distributed Databases & Query Engines',
  //       Svg: require('@site/static/img/green-dd.svg').default,
  //       description: (
  //           <>
  //               Constructing High-Performance, Scalable Distributed Databases and Query Engines with Cassandra, Trino, Presto, and More
  //           </>
  //       ),
  //   },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} style={{width: '40%'}} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h2><b>{title}</b></h2>
        <p>{description}</p>
      </div>
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

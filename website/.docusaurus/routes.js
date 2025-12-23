import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '539'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '907'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', '233'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '132'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', 'c50'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '3bc'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', 'df0'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', 'd55'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', '9d3'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms',
        component: ComponentCreator('/algorithms', 'fcf'),
        exact: true
      },
      {
        path: '/algorithms/',
        component: ComponentCreator('/algorithms/', 'c9a'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/ant_colony',
        component: ComponentCreator('/algorithms/ant_colony', '16a'),
        exact: true
      },
      {
        path: '/algorithms/ant-colony-optimization',
        component: ComponentCreator('/algorithms/ant-colony-optimization', 'fdf'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/bayesian',
        component: ComponentCreator('/algorithms/bayesian', 'f95'),
        exact: true
      },
      {
        path: '/algorithms/bayesian-optimization',
        component: ComponentCreator('/algorithms/bayesian-optimization', 'a31'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/genetic',
        component: ComponentCreator('/algorithms/genetic', '71c'),
        exact: true
      },
      {
        path: '/algorithms/genetic-algorithm',
        component: ComponentCreator('/algorithms/genetic-algorithm', '680'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/grey_wolf',
        component: ComponentCreator('/algorithms/grey_wolf', 'f5c'),
        exact: true
      },
      {
        path: '/algorithms/grey-wolf-optimization',
        component: ComponentCreator('/algorithms/grey-wolf-optimization', '8f9'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/grid_search',
        component: ComponentCreator('/algorithms/grid_search', '5e1'),
        exact: true
      },
      {
        path: '/algorithms/grid-search',
        component: ComponentCreator('/algorithms/grid-search', 'b99'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/pso',
        component: ComponentCreator('/algorithms/pso', '8f9'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/random_search',
        component: ComponentCreator('/algorithms/random_search', '191'),
        exact: true
      },
      {
        path: '/algorithms/random-search',
        component: ComponentCreator('/algorithms/random-search', 'd47'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/simulated_annealing',
        component: ComponentCreator('/algorithms/simulated_annealing', '429'),
        exact: true
      },
      {
        path: '/algorithms/simulated-annealing',
        component: ComponentCreator('/algorithms/simulated-annealing', 'cc2'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/tpe',
        component: ComponentCreator('/algorithms/tpe', '8ee'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/api',
        component: ComponentCreator('/api', 'ef7'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/archives/removed_examples_custom_metric',
        component: ComponentCreator('/archives/removed_examples_custom_metric', '303'),
        exact: true
      },
      {
        path: '/contributing',
        component: ComponentCreator('/contributing', 'd27'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/design-system',
        component: ComponentCreator('/design-system', '9b9'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/',
        component: ComponentCreator('/docs/', 'c45'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/examples',
        component: ComponentCreator('/examples', '969'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/examples_custom_metric',
        component: ComponentCreator('/examples_custom_metric', '8f1'),
        exact: true
      },
      {
        path: '/getting-started',
        component: ComponentCreator('/getting-started', 'f3c'),
        exact: true,
        sidebar: "docsSidebar"
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];

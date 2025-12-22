import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/',
    component: ComponentCreator('/', 'c6c'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', '9d3'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms',
        component: ComponentCreator('/algorithms', 'd65'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/',
        component: ComponentCreator('/algorithms/', 'c9a'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/ant_colony',
        component: ComponentCreator('/algorithms/ant_colony', '71f'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/bayesian',
        component: ComponentCreator('/algorithms/bayesian', 'ce1'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/genetic',
        component: ComponentCreator('/algorithms/genetic', '427'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/grey_wolf',
        component: ComponentCreator('/algorithms/grey_wolf', '77f'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/grid_search',
        component: ComponentCreator('/algorithms/grid_search', 'f3f'),
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
        component: ComponentCreator('/algorithms/random_search', '57c'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/algorithms/simulated_annealing',
        component: ComponentCreator('/algorithms/simulated_annealing', '52e'),
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
        component: ComponentCreator('/examples_custom_metric', '825'),
        exact: true,
        sidebar: "docsSidebar"
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

// website/sidebars.js
// Custom sidebar for OptiFlowX documentation
module.exports = {
  docsSidebar: [
    'index',
    'getting-started',
    {
      type: 'category',
      label: 'Algorithms',
      items: [
        'algorithms/index',
        'algorithms/pso',
        'algorithms/simulated-annealing',
        'algorithms/bayesian-optimization',
        'algorithms/tpe',
        'algorithms/genetic-algorithm',
        'algorithms/random-search',
        'algorithms/grid-search',
        'algorithms/grey-wolf-optimization',
        'algorithms/ant-colony-optimization',
      ],
    },
    'examples',
    'api',
    'design-system',
    'contributing',
    {
      type: 'category',
      label: 'Internal notes',
      items: [
        'docs/index',
      ],
    },
  ],
};

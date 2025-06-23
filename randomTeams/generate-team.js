// generate-team.js

const { Teams } = require('../pokemon-showdown/dist/sim');

function getRandomSeed() {
  return Array.from({ length: 4 }, () => Math.floor(Math.random() * 0x10000));
}


async function main() {
  const format = 'gen9randombattle'; // or any supported format ID
  const seed = getRandomSeed();
  const options = { seed };

  const team = await Teams.generate(format, options);
  const exportedTeam = Teams.export(team);
  console.log(exportedTeam);
}

main();

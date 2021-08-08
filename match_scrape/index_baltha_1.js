const superagent = require('superagent');
fs = require('fs');

const API_KEY = '26b64ef5a99c496897c9d5eba705b83a';
const SEASON = 2020;

let requestRes;
superagent
  .get(`api.football-data.org/v2/competitions/FL1/matches?season=${SEASON}`)
  .set('X-Auth-Token', API_KEY)
  .end((err, res) => {
    if (err) {
      console.log(err);
      return;
    }
    requestRes = res.body;
    console.log(requestRes.matches[0]);
    requestRes.matches.forEach((element) => {
      let elementResult = '';
      elementResult =
        elementResult +
        element.homeTeam.name +
        ' ' +
        element.awayTeam.name +
        ' ';
      if (element.status === 'FINISHED') {
        elementResult += element.score.fullTime.homeTeam + ' ' + element.score.fullTime.awayTeam + ' ' ;
      } else {
        elementResult = elementResult + 'None' + ' ';
      }

      elementResult = elementResult + 'None None None' + ' ';
      elementResult = elementResult + element.utcDate;
      elementResult = elementResult + '\n';
      fs.appendFile(`result_${SEASON}.txt`, elementResult, function (err) {
        if (err) return console.log(err);
      });
    });
  });

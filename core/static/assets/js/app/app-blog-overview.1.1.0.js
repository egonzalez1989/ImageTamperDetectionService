/*
 |--------------------------------------------------------------------------
 | Shards Dashboards: Blog Overview Template
 |--------------------------------------------------------------------------
 */

'use strict';


/*
(function ($) {
  $(document).ready(function () {
    //
    // Image 1 histogram
    //

    var bouCtx1 = document.getElementById('chart1').getContext('2d');
    var hist = JSON.parse(document.getElementById('hist1').value);

    // Data
    var bouData = {
      // Generate the days labels on the X axis.
      labels: hist['Y'],
      datasets: [{
        //label: 'Feature value',
        fill: 'start',
        data: hist['X'],
        backgroundColor: 'rgba(255,0,0,0.1)',
        borderColor: 'rgba(255,0,0,1)',
        pointBackgroundColor: '#ffffff',
        pointHoverBackgroundColor: 'rgb(255,0,0)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3
      }]};

    // Options
    var bouOptions = {
      responsive: true,
      legend: {
        position: 'top'
      },
      elements: {
        line: {
          // A higher value makes the line look skewed at this ratio.
          tension: 0.3
        },
        point: {
          radius: 0
        }
      },
      scales: {
        xAxes: [{
          gridLines: false,
          ticks: {
            callback: function (tick, index) {
              // Jump every 7 values on the X axis labels to avoid clutter.
              return index % 10 !== 0 ? '' : tick.toFixed(3);
            }
          }
        }],
        yAxes: [{
          ticks: {
            suggestedMax: Math.max.apply(Math, hist['X']),
            callback: function (tick, index, ticks) {
              if (tick === 0) {
                return tick;
              }
              // Format the amounts using Ks for thousands.
              return tick > 999 ? (tick/ 1000).toFixed(1) + 'K' : tick.toFixed(2);
            }
          }
        }]
      },
      // Uncomment the next lines in order to disable the animations.
      // animation: {
      //   duration: 0
      // },
      hover: {
        mode: 'x',
        intersect: false
      },
      tooltips: {
        custom: false,
        mode: 'nearest',
        intersect: false
      },
      //events: ['click'],
    };

    // Generate the Analytics Overview chart.
    chart1 = new Chart(bouCtx1, {
      type: 'LineWithLine',
      data: bouData,
      options: bouOptions
    });


    window.BlogOverviewUsers = chart1;








    //
    // Image 2 histogram
    //

    var bouCtx2 = document.getElementsByClassName('blog-overview-users')[1];
    var hist = JSON.parse(document.getElementById('hist2').value);
    // Data
    var bouData = {
      // Generate the days labels on the X axis.
      labels: hist['Y'],
      datasets: [{
        //label: 'Feature value',
        fill: 'start',
        data: hist['X'],
        backgroundColor: 'rgba(0,255,0,0.1)',
        borderColor: 'rgba(0,255,0,1)',
        pointBackgroundColor: '#ffffff',
        pointHoverBackgroundColor: 'rgb(0,255,0)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3
      }]};

    // Options
    var bouOptions = {
      responsive: true,
      legend: {
        position: 'top'
      },
      elements: {
        line: {
          // A higher value makes the line look skewed at this ratio.
          tension: 0.3
        },
        point: {
          radius: 0
        }
      },
      scales: {
        xAxes: [{
          gridLines: false,
          ticks: {
            callback: function (tick, index) {
              // Jump every 7 values on the X axis labels to avoid clutter.
              return index % 10 !== 0 ? '' : tick;
            }
          }
        }],
        yAxes: [{
          ticks: {
            suggestedMax: Math.max.apply(Math, hist['X']),
            callback: function (tick, index, ticks) {
              if (tick === 0) {
                return tick;
              }
              // Format the amounts using Ks for thousands.
              return tick > 999 ? (tick/ 1000).toFixed(1) + 'K' : tick.toFixed(3);
            }
          }
        }]
      },
      // Uncomment the next lines in order to disable the animations.
      // animation: {
      //   duration: 0
      // },
      hover: {
        mode: 'nearest',
        intersect: false
      },
      tooltips: {
        custom: false,
        mode: 'nearest',
        intersect: false
      },
      events: ['click'],
    };

    // Generate the Analytics Overview chart.
    window.BlogOverviewUsers = new Chart(bouCtx2, {
      type: 'LineWithLine',
      data: bouData,
      options: bouOptions
    });













    //
    // Image 3 histogram
    //

    var bouCtx3 = document.getElementsByClassName('blog-overview-users')[2];
    var hist = JSON.parse(document.getElementById('hist3').value);
    // Data
    var bouData = {
      // Generate the days labels on the X axis.
      labels: hist['Y'],
      datasets: [{
        //label: 'Feature value',
        fill: 'start',
        data: hist['X'],
        backgroundColor: 'rgba(0,0,255,0.1)',
        borderColor: 'rgba(0,0,255,1)',
        pointBackgroundColor: '#ffffff',
        pointHoverBackgroundColor: 'rgb(0,0,255)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3
      }]};

    // Options
    var bouOptions = {
      responsive: true,
      legend: {
        position: 'top'
      },
      elements: {
        line: {
          // A higher value makes the line look skewed at this ratio.
          tension: 0.3
        },
        point: {
          radius: 0
        }
      },
      scales: {
        xAxes: [{
          gridLines: false,
          ticks: {
            callback: function (tick, index) {
              // Jump every 7 values on the X axis labels to avoid clutter.
              return index % 10 !== 0 ? '' : tick;
            }
          }
        }],
        yAxes: [{
          ticks: {
            suggestedMax: Math.max.apply(Math, hist['X']),
            callback: function (tick, index, ticks) {
              if (tick === 0) {
                return tick;
              }
              // Format the amounts using Ks for thousands.
              return tick > 999 ? (tick/ 1000).toFixed(1) + 'K' : tick.toFixed(3);
            }
          }
        }]
      },
      // Uncomment the next lines in order to disable the animations.
      // animation: {
      //   duration: 0
      // },
      hover: {
        mode: 'nearest',
        intersect: false
      },
      tooltips: {
        custom: false,
        mode: 'nearest',
        intersect: false
      }
    };

    // Generate the Analytics Overview chart.
    window.BlogOverviewUsers = new Chart(bouCtx3, {
      type: 'LineWithLine',
      data: bouData,
      options: bouOptions
    });
  });
})(jQuery);
*/
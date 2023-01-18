var RED_INTENCITY_COEF = 0.2126;
var GREEN_INTENCITY_COEF = 0.7152;
var BLUE_INTENCITY_COEF = 0.0722;

const fmaps = [new Image(), new Image(), new Image()];
var imgctx = document.getElementById('canvas0').getContext('2d');
var img = document.getElementById('original_image');
img.src = document.getElementById('original_image').getAttribute('src');
imgctx.drawImage(img, 0, 0);

function getImageArray(canvasid) {
    var cnv = document.getElementById(canvasid);
    var ctx = cnv.getContext('2d');
    var imageData = ctx.getImageData(0, 0, cnv.width, cnv.height);
    return imageData.data;
}

// Masked tampered image
function tamper_mask() {
    img.src = document.getElementById('original_image').getAttribute('src');
    imgctx.drawImage(img, 0, 0);
    var w = img.width, h = img.height;
    var imageData = imgctx.getImageData(0, 0, w, h);
    var data = imageData.data;
    var chk1 = document.getElementById('chk1').checked;
    var chk2 = document.getElementById('chk2').checked;
    var chk3 = document.getElementById('chk3').checked;
    if (chk1 || chk2 || chk3) {
        // Zero matrix
        data1 = getImageArray('canvas1');
        data2 = getImageArray('canvas2');
        data3 = getImageArray('canvas3');
        var paint = 0;
        for (var i = 0; i < data.length; i += 4) {
            paint = (chk1 ? data1[i] /255 : 0) + (chk2 ? data2[i]/255 : 0) + (chk3 ? data3[i]/255 : 0);
            if (paint == 0) {
                data[i] = data[i];
            } else {
                data[i] = Math.floor((255 + data[i]) / 2);
                data[i+1] = Math.floor(data[i+1] / 2);
                data[i+2] = Math.floor(data[i+2] / 2);
            }
        }
    }
    // overwrite original image
    imgctx.putImageData(imageData, 0, 0);
}

// Binarize
function binarize(th, ctx, w, h) {
    var imageData = ctx.getImageData(0, 0, w, h);
    var data = imageData.data;
    var val;
    for(var i = 0; i < data.length; i += 4) {
        var brightness = RED_INTENCITY_COEF * data[i] + GREEN_INTENCITY_COEF * data[i + 1] + BLUE_INTENCITY_COEF * data[i + 2];
        val = ((brightness > th) ? 0 : 1);
        data[i] = 255 * val;
        data[i + 1] = 255 * val;
        data[i + 2] = 255 * val;
    }
    // overwrite original image
    var d = new Date();
    var n = d.getMilliseconds();
    ctx.putImageData(imageData, 0, 0);
    return data;
};

// Load each image
function img_loader(idx) {
    var cnv = document.getElementById('canvas' + idx);
    var ctx = cnv.getContext('2d');
    var src = document.getElementById('channel' + idx).getAttribute('src');
    var idx0 = idx-1;
    var fmap = fmaps[idx0];
    fmap.src = src;
    fmap.onload = function() {
        var apply_sgmt = document.getElementById('chk' + idx).checked ;
        var w = fmap.width, h = fmap.height;
        cnv.height = h;
        cnv.width = w;
        ctx.drawImage(fmap, 0, 0);
        if (apply_sgmt === true) {
            //toGrayscale(ctx, w, h);
           var th = document.getElementById('th'+idx).getAttribute('value');
           var hist = JSON.parse(document.getElementById('hist'+idx).value);
           var min = Math.min.apply(null, hist['Y']);
           var max = Math.max.apply(null, hist['Y']);
           th = 255 * (th - min) / (max - min);
           binarize(th, ctx, w, h);
        } else {
            ctx.putImageData(ctx.getImageData(0, 0, w, h), 0, 0);
        }
        tamper_mask();
    };
};

function chart_loader(idx, color) {
  var input = document.getElementById('chk' + idx);
  var ctx = document.getElementById('chart' + idx).getContext('2d');
  var hist = JSON.parse(document.getElementById('hist' + idx).value);
  var idx0 = idx-1;
  input.addEventListener('change', function () {
      fmaps[idx0].src = document.getElementById('channel' + idx).getAttribute('src');
  });
  var bouData = {
      // Generate the days labels on the X axis.
      labels: hist['Y'],
      datasets: [{
        //label: 'Feature value',
        fill: 'start',
        data: hist['X'],
        backgroundColor: 'rgba(' + color + ',0.1)',
        borderColor: 'rgba(' + color + ',1)',
        pointBackgroundColor: '#ffffff',
        pointHoverBackgroundColor: 'rgb(' + color + ')',
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
              return index % 10 !== 0 ? '' : tick.toFixed(2);
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
    var chart = new Chart(ctx, {
      type: 'LineWithLine',
      data: bouData,
      options: bouOptions
    });

    window.BlogOverviewUsers = chart;

    var featctx = document.getElementById('canvas' + idx).getContext('2d');
    document.getElementById("chart" + idx).onclick = function(evt) {
      var activePoints = chart.getElementsAtEventForMode(evt, 'x', chart.options);
      var firstPoint = activePoints[0];
      var th = chart.data.labels[firstPoint._index];
      var w = fmaps[idx0].width, h = fmaps[idx0].height;
      document.getElementById('th' + idx).setAttribute('value', th);
      fmaps[idx0].src = document.getElementById('channel' + idx).getAttribute('src');
      var apply_sgmt = document.getElementById('chk' + idx).checked;
      if (apply_sgmt) {
          var hist = JSON.parse(document.getElementById('hist'+idx).value);
          var min = Math.min.apply(null, hist['Y']);
          var max = Math.max.apply(null, hist['Y']);
          th = 255 * (th - min) / (max - min);
          binarize(th, featctx, w, h);
      }
      tamper_mask();
  }
}


(function ($) {
  $(document).ready(function () {
    chart_loader(1, '255,0,0');
    chart_loader(2, '0,255,0');
    chart_loader(3, '0,0,255');
    });
})(jQuery);

// First image
// Preparing image and histogram
img_loader(1);
img_loader(2);
img_loader(3);
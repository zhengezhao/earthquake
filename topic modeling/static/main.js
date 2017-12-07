var eq = [];
var eq_query=[];
var start = 0;
var end =0;
var changed_num = "";
var totalnum = '{{totalnum|safe}}';
var options = [];
for (var i = 1; i <= parseInt(50); i++) {
  options.push(i);
}

d3.selection.prototype.callReturn = function(callable)
{
      return callable(this);
};

 var color = d3.scaleOrdinal(d3.schemeCategory10);

var xScale= d3.scaleLinear().range([0,500]);


var select = document.getElementById("selectNumber");

for(var i = 0; i < options.length; i++) {
  var opt = options[i];
  var el = document.createElement("option");
  el.textContent = opt;
  el.value = opt;
  select.appendChild(el);
}

function changenum(){
  var e = document.getElementById("selectNumber");
  changed_num = e.options[e.selectedIndex].value;
  d3.request("http://0.0.0.0:5050/data")
    .header("X-Requested-With", "XMLHttpRequest")
    .header("Content-Type", "application/x-www-form-urlencoded")
    .post(JSON.stringify(changed_num), function(d)
    {
      eq= JSON.parse(d.response);
      var maxwidth = d3.max(eq, d=>d.j);
      xScale.domain([0,maxwidth]);

      d3.select("#eqplot").selectAll("rect").remove();
      d3.select("#eqplot")
        .callReturn(createRects)
        .attr("x", function(d) { console.log(xScale(d.j), Math.floor(xScale(d.j))); return Math.floor(xScale(d.j)); })
        .attr("width", Math.ceil(500.0/maxwidth*1.05) ) ;

      var brush = d3.brushX()
        .extent([[0,0],[500,110]])
        .on("brush end", brushed);

      d3.select("#eqplot").append("g")
        .attr("class", "brush")
        .call(brush);

      function brushed(){
        var s = d3.event.selection;
        if(s){
          start =  Math.floor(xScale.invert(s[0]));
          end = Math.floor(xScale.invert(s[1]));
        }
      }
    });

}


function find_NN(){

  var dict={
      "start": start,
      "end": end,
      "eqnum": changed_num};

  d3.request("http://0.0.0.0:5050/data_query")
    .header("X-Requested-With", "XMLHttpRequest")
    .header("Content-Type", "application/x-www-form-urlencoded")
    .post(JSON.stringify(dict), function(d){

      eq_query = JSON.parse(d.response);

      for(var i =0; i<eq_query.length ;i++){
        var eqid = "rank" + i.toString();
        var eq_d = eq_query[i]["data"];
        var eq_n = eq_query[i]["eqnum"];
        var eq_value = eq_query[i]["value"];
        //console.log(eq_d);
        var maxwidth = d3.max(eq_d, d=>d.j);
        //console.log(maxwidth);
        var x = d3.scaleLinear().range([0,xScale(end)-xScale(start)+1 ]).domain([0,maxwidth]);
        d3.select("#"+eqid).selectAll("rect").remove();
        d3.select("#"+eqid).selectAll("text").remove();

        d3.select("#"+eqid)
        .append("text")
        .attr("x", 300)
        .attr("y", 50 )
        .attr("text-anchor", "middle")
        .style("font-size", "20px")
        .style("text-decoration", "underline")
        .text("earthquake:"+ eq_n.toString()+"  difference:"+ eq_value.toFixed(4));

        d3.select("#"+eqid)
          .append("g")
          .selectAll("rect")
          .data(eq_d)
          .enter()
          .append("rect")
          .attr("y", function(d) { return 10+d.i*20; })
          .attr("height", 20)
          .attr("fill",function(d){return colorScales[d.i](d.value);})
        //  .attr("opacity",function(d){return d.value;})
          .attr("x", function(d) { return Math.floor(x(d.j)); })
          .attr("width", Math.ceil((xScale(end)-xScale(start))/maxwidth*1.05) ) ;



      }
  });
}

function createSvg(sel)
  {
    return sel
      .append("svg")
      .attr("id", "eqplot")
      .attr("width", svgWidth)
      .attr("height", svgHeight);
  }

var colorScales = d3.range(5).map(function(i) {
  return d3.scaleLinear().domain([0, 1]).range(["white", color(i)]);
});

function createRects(sel)
  {
    return sel
        .append("g")
        .selectAll("rect")
        .data(eq)
        .enter()
        .append("rect")
        .attr("y", function(d) { return 10+d.i*20; })
        .attr("height", 20)
        .attr("fill",function(d){return colorScales[d.i](d.value);/*color(d.i)*/});
        //.attr("opacity",function(d){return d.value;});
  }

var svgHeight = 110;
var svgWidth =510;

$(document).ready(function(){
  d3.select("#show_eq")
  .callReturn(createSvg)
  .callReturn(createRects);

  for(var i =0; i<10; i++){
    eqid = "rank"+i.toString() ;
  d3.select("#show_eq_query")
    .append("svg")
    .attr("id", eqid)
    .attr("width", svgWidth)
    .attr("height", svgHeight);
  }


});




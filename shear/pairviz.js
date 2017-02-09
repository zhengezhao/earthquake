//data1 = data1.slice(0,4000);
//data2 = data2.slice(0,100);
var filterdata = function (data){
    var dataTemp = [];
    for(var i=0; i<data.length;i++)
        {if ((i%5)===0)
            {
                dataTemp.push(data[i]);
               }
           }
    return dataTemp;           
};

data1 =filterdata(data1);
data2 =filterdata(data2);
var scalex=d3.scaleLinear().domain(d3.extent(data1.concat(data2),function(d){return d.v[0];})).range([10,290]);
var scaley=d3.scaleLinear().domain(d3.extent(data1.concat(data2),function(d){return d.v[1];})).range([10,290]);
var scalepathx=d3.scaleLinear().domain([-0.35,0.35]).range([10,290]);
var scalepathy=d3.scaleLinear().domain([0,13]).range([290,10]);

var linefunction=d3.line()
	.x(function(d,i){return scalepathx(d);})
	.y(function(d,i){return scalepathy(i);});

d3.select("#plot1").append("g").selectAll("circle").data(data1).enter().append("circle")
    .attr("cx",function(d){return scalex(d.v[0]);})
    .attr("cy",function(d){return scaley(-d.v[1]);}).attr("r",3).attr("fill","RGBA(0,0,255,0.5)");

d3.select("#plot2").append("g").selectAll("circle").data(data2).enter().append("circle")
    .attr("cx",function(d){return scalex(d.v[0]);})
    .attr("cy",function(d){return scaley(-d.v[1]);}).attr("r",3).attr("fill","RGBA(0,0,255,0.5)");

var pc1 = [ 0.54524666,  0.50131919,  0.43259372,  0.34275148,  0.23873171,  0.12975694,
            0.02621332, -0.06148233, -0.1238945,  -0.15334199, -0.1443423,  -0.0936698,   0        ];
var pc2 = [-0.0219984,   0.02151634,  0.08465168,  0.16193513,  0.24503503,  0.32382765,
           0.38756593,  0.42599244,  0.43024757,  0.39346067,  0.3109671,   0.18015548,  -0.        ];

//var data3 = [];
//for (var i=0; i<data1.length; i+= 200) {
//   data3.push(data1[i]);
//}

//var data4 = [];
//for (var i=0; i<data1.length; i+= 200) {
//    data4.push(data2[i]);
//}

d3.select("#plot3")
    .append("g")
    .selectAll("path")
    .data(data1)
    .enter()
    .append("path")
    .attr("stroke","blue")
    .attr("stroke-width",1)
    .attr('fill','none')
    .attr("d", function(d) {
        return linefunction(d.vCap);
    });

d3.select("#plot4")
    .append("g")
    .selectAll("path")
    .data(data2)
    .enter()
    .append("path")
    .attr("stroke","blue")
    .attr("stroke-width",1)
    .attr('fill','none')
    .attr("d", function(d) {
        return linefunction(d.vCap);
    });




var cursorCurveData = {vs: [0,0,0,0,0,0,0,0,0,0,0,0,0]};
var cursorCurve1 = d3.select("#plot3").append("g")
        .append("path")
        .datum(cursorCurveData)
        .attr("stroke", "red")
        .attr("stroke-width", "3")
        .attr("fill", "none");

var cursorCurve2 = d3.select("#plot4").append("g")
        .append("path")
        .datum(cursorCurveData)
        .attr("stroke", "red")
        .attr("stroke-width", "3")
        .attr("fill", "none");


function updateCursorCurve(cursorCurve) {
    cursorCurve
        .attr("d", function(d) {
            return linefunction(d.vs);
        });
}

updateCursorCurve(cursorCurve1);
updateCursorCurve(cursorCurve2);	

function makeHoverCurveWidget(sel,cursorCurve)
{
    sel.append("g")
    .append("rect")
    .attr("fill", "rgba(0,0,0,0)") // ugly hack
    .attr("width", 300).attr("height", 300)
    .on("mousemove", function() {
        var x = d3.event.offsetX,
            y = d3.event.offsetY;
        var pc1Weight = scalex.invert(x),
            pc2Weight = scaley.invert(y);
        for (var i=0; i<13; ++i) {
            cursorCurveData.vs[i] =
                pc1Weight * pc1[i] +
                pc2Weight * pc2[i];
        }
        updateCursorCurve(cursorCurve);
    });

    sel.append("g")
        .append("line")
        .attr("x1", scalex(-1))
        .attr("x2", scalex(1))
        .attr("y1", scaley(0))
        .attr("y2", scaley(0))
        .attr("stroke", "black");

    sel.append("g")
        .append("line")
        .attr("x1", scalex(0))
        .attr("x2", scalex(0))
        .attr("y1", scaley(-1))
        .attr("y2", scaley(1))
        .attr("stroke", "black");
}

makeHoverCurveWidget(d3.select("#plot1"),cursorCurve1);
makeHoverCurveWidget(d3.select("#plot2"),cursorCurve2);


var myb1 = d3.brush()
	.on("brush end", updateb1);

var myb2 = d3.brush()
	.on("brush end", updateb2);
//d3.select('#plot1').call(myb1);
//d3.select('#plot2').call(myb2);
//b1 and b2 are selected region, in pixel coord inside each svg
var b1,b2;
function updateb1(){
    b1 = d3.event.selection;
	onBrush("#plot3");
}

function updateb2(){
    b2 = d3.event.selection;
	onBrush("#plot4");
}

function onBrush(plot){
	var allPaths = d3.select(plot).selectAll("path");

    if (!b1 && !b2) {
		allPaths.attr("stroke", 'rgba(0,0,255,0.5)');
		return;
    }

    function isSelected(d) {
		//given a data point d, return whether it is inside the selected region(s)
		if (!b1 && !b2){ return false;}
		var result = true;

		if (b1 && d.v){
			result = result 
					&& b1[0][0] <= scalex(d.v[0])
					&& b1[1][0] >= scalex(d.v[0])
					&& b1[0][1] <= scaley(d.v[1])
					&& b1[1][1] >= scaley(d.v[1]);
		}
		return result;	
    }
    
	//change the attr of selected / non-selecte
	var allPathsfilter = allPaths.filter(isSelected);
	var allPathsfilterdata = allPaths.filter(isSelected).data();
	//console.log(allPathsfilterdata);
	allPathsfilter.remove();
    allPaths.filter(isSelected)
		.attr("stroke", "rgba(255,0,255,0.2)");
	allPaths.filter(function(d){ return !isSelected(d)})
		.attr("stroke", "rgba(0,0,255,0.2)");

    d3.select("#plot3").selectAll("x").data(allPathsfilterdata).enter()
    .append("path")
    .attr("stroke","red")
    .attr("stroke-width",1)
   .attr('fill','none')
   .attr("d", function(d) {
        return linefunction(d.vCap);
    });
}



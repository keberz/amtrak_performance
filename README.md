# fra-amtrak: Federal Railroad Administration Amtrak Quarterly Performance Metrics

## Introduction

The U.S. [Department of Transportation](https://www.transportation.gov/) (DOT),
[Federal Railroad Administration](https://railroads.dot.gov/) (FRA) publishes
[quarterly reports](https://railroads.dot.gov/rail-network-development/passenger-rail/amtrak/intercity-passenger-rail-service-quality-and)
on the performance and service quality of intercity passenger rail service. The
reports include metrics on customer on-time performance, host running time,
station performance, and train delays. The data is collected from Amtrak and
host railroads and is used to evaluate the performance of intercity passenger
rail service.

## Acronyms

* AMTK: Amtrak reporting mark
* BTS: Bureau of Transportation Statistics
* CFR: Code of Federal Regulations
* DOT: U.S. Department of Transportation
* FRA: Federal Railroad Administration
* FY: Fiscal Year
* NEC: Northeast Corridor
* OTP: On-Time Performance
* Q: Fiscal Quarter

## Definitions

### [CFR 49, 273.3](https://www.ecfr.gov/current/title-49/section-273.3) [Amtrak-responsible delays](https://www.ecfr.gov/current/title-49/part-273#p-273.3(Amtrak-responsible%20delays))

> means delays recorded by Amtrak, in accordance with Amtrak procedures, as Amtrak-responsible
> delays, including passenger-related delays at stations, Amtrak equipment failures, holding for
> connections, injuries, initial terminal delays, servicing delays, crew and system delays, and
> other miscellaneous Amtrak-responsible delays.

### [CFR 49, 273.3](https://www.ecfr.gov/current/title-49/section-273.3) [Host railroad](https://www.ecfr.gov/current/title-49/part-273#p-273.3(Host%20railroad))

> means a railroad that is directly accountable to Amtrak by agreement for Amtrak operations over a
> railroad line segment. Amtrak is a host railroad of Amtrak trains and other trains operating over
> an Amtrak owned or controlled railroad line segment. For purposes of the certified schedule metric
> under § [273.5(c)](https://www.ecfr.gov/current/title-49/section-273.5#p-273.5(c)), Amtrak is not
> a host railroad.

### [CFR 49, 273.3](https://www.ecfr.gov/current/title-49/section-273.3) [Host-responsible delays](https://www.ecfr.gov/current/title-49/part-273#p-273.3(Host-responsible%20delays))

> means delays recorded by Amtrak, in accordance with Amtrak procedures, as host-responsible delays,
> including freight train interference, slow orders, signals, routing, maintenance of way, commuter
> train interference, passenger train interference, catenary or wayside power system failure, and
> detours.

### [CFR 49, 273.3](https://www.ecfr.gov/current/title-49/section-273.3) [Third party delays](https://www.ecfr.gov/current/title-49/part-273#p-273.3(Third%20party%20delays))

> means delays recorded by Amtrak, in accordance with Amtrak procedures, as third party delays,
> including bridge strikes, debris strikes, customs, drawbridge openings, police-related delays,
> trespassers, vehicle strikes, utility company delays, weather-related delays (including heat or
> cold orders, storms, floods/washouts, earthquake-related delays, slippery rail due to leaves,
> flash-flood warnings, wayside defect detector actuations caused by ice, and high-wind restrictions),
> acts of God, or waiting for scheduled departure time.

### [49 CFR 273.5(a)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(a)) Customer on-time performance

* #### [49 CFR 273.5(a)(1)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(a)(1)) Metric

   > The customer on-time performance metric is the percentage of all customers on an
   > intercity passenger rail train who arrive at their detraining point no later than `15` minutes
   > after their published scheduled arrival time, reported by train and by route.

* #### [49 CFR 273.5(a)(2)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(a)(2)) Standard

   > The customer on-time performance minimum standard is `80` percent for any `2`
   > consecutive calendar quarters.

### [49 CFR 273.5(g)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(g)) Host running time

> The host running time metric is the average actual running time and the median actual running
> time compared with the scheduled running time between the first and final reporting points for a
> host railroad set forth in the Amtrak schedule skeleton, reported by route, by train, and by
> host railroad (excluding switching and terminal railroads).

### [49 CFR 273.5(f)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(f)) Station performance

> The station performance metric is the number of detraining passengers, the number of late
> passengers, and the average minutes late that late customers arrive at their detraining stations,
> reported by route, by train, and by station. The average minutes late per late customer
> calculation excludes on-time customers that arrive no later than 15 minutes after their
> scheduled time.

### [49 CFR 273.5(d)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(d)) Train delays

> The train delays metric is the minutes of delay for all Amtrak-responsible delays,
> host-responsible delays, and third party delays, for the host railroad territory within each
> route. The train delays metric is reported by delay code by: total minutes of delay;
> Amtrak-responsible delays; Amtrak's host-responsible delays; Amtrak's host responsible delays
> and Amtrak-responsible delays, combined; non-Amtrak host-responsible delays; and third party
> delays. The train delays metric is also reported by the number of non-Amtrak host-responsible
> delay minutes disputed by host railroad and not resolved by Amtrak.

### [49 CFR 273.5(e)](https://www.ecfr.gov/current/title-49/part-273#p-273.5(e)) Train delays per `10,000` train miles

> The train delays per `10,000` train miles metric is the minutes of delay per `10,000` train
> miles for all Amtrak-responsible and host-responsible delays, for the host railroad territory
> within each route.

## Modules

The notebooks rely on a number of Python modules to process, analyze, and
visualize the data. The modules are found in the `fra-amtrak` directory.

Some modules feature functions that have yet to be implemented. Some function
blocks may only be partially implemented and could trigger runtime exceptions
if not corrected. Your job as a member of the team is to address these issues
and ensure that the modules perform as expected.

Review the function's docstring to better understand the task it is to perform,
the parameters it defines, and the return value it computes.

## Visualizations

The [Vega-Altair](https://altair-viz.github.io/) library is used to visualize
the data. The Vega-Altair data model favors tabular data in the guise of a
[pandas](https://pandas.pydata.org/) `DataFrame` but generating charts from
JSON objects and Python dictionaries are also supported.

:bulb: You are not expected to learn Vega-Altair or generate charts for this
assignment. That said, learning how to visualize data data in a way that is
both expressive and effective is a valuable skill and the teaching team
encourages you to explore Vega-Altair or other visualization libraries such as
[Bokeh](https://github.com/bokeh/bokeh),
[Matplotlib](https://github.com/matplotlib/matplotlib),
[Plotly](https://github.com/plotly/plotly.py),
and [Seaborn](https://github.com/mwaskom/seaborn).

## TOML

The file `notbook.toml` contains notebook constants. [TOML](https://toml.io/en/)
is a data serialization language that is easy to read due to its minimal syntax.
The TOML file is loaded into the notebook as a Python dictionary using the
[tomllib](https://docs.python.org/3/library/tomllib.html) module (introduced in
Python `3.11`).

Favor the use of constants in your work over hard-coded strings.

## Watermark

Each notebook utilizes the iPython [watermark](https://github.com/rasbt/watermark)
magic extension to document the runtime environment (Python, iPython, and
package dependency versions), operating system, CPU, and other hardware
information as an aid to reproducing the computational environment.

## Sources

U.S. Census Bureau, ["Geographic Levels"](https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html).

U.S. Department of Transportation (DOT). Bureau of Transportation Statistics (BTS).
[Amtrak Stations](https://geodata.bts.gov/datasets/1ed62a9f46304679aaa396bed4c8565a_0/about).

U.S. Department of Transportation (DOT). Federal Railroad Administration (FRA).
[_Intercity Passenger Rail Service Quality and Performance Reports_](https://railroads.dot.gov/rail-network-development/passenger-rail/amtrak/intercity-passenger-rail-service-quality-and),
quarterly reports.

U.S. Department of Transportation (DOT). Federal Railroad Administration (FRA).
[_Methodology Report for the Performance and Service Quality of Intercity Passenger Train Operations_](https://railroads.dot.gov/sites/fra.dot.gov/files/2024-08/Methodology%20Report_FY24Q3_web.pdf),
Fiscal Year 2024, v.2.

U.S. National Archives, Code of Federal Regulations, Title 49, Subtitle B, Chapter II,
[Part 273](https://www.ecfr.gov/current/title-49/subtitle-B/chapter-II/part-273).

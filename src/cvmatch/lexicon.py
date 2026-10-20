"""Лексикон навичок. Групи синонімів та споріднених навичок перенесено з конфігурації методу
CVMaster (method-config.json, бакалаврська робота автора) і розширено типовими технологіями ПЗ."""
SYNONYMS = [["javascript","js"],["typescript","ts"],["node.js","node","nodejs"],["react","react.js","reactjs"],
 ["vue","vue.js","vuejs"],["angular","angular.js","angularjs"],["next.js","nextjs"],["nestjs","nest.js"],
 ["postgresql","postgres"],["mongodb","mongo"],["kubernetes","k8s"],["aws","amazon web services"],
 ["gcp","google cloud platform"],["rest api","rest","restful","restful api"],
 ["ci/cd","cicd","continuous integration","continuous delivery"],["qa","quality assurance"],
 [".net",".net core","dotnet","asp.net","asp.net core","asp.net mvc"],["c#","csharp","c sharp"],
 ["sql server","mssql","ms sql","microsoft sql server"],["spring","spring boot","springboot","spring mvc"],
 ["html","html5"],["css","css3"],["oop","object oriented","object-oriented"],["git","github","gitlab"],
 ["agile","scrum"],["unit testing","unit test","unit tests","junit","nunit","xunit"],["c++","cpp"]]
RELATED = [["javascript","typescript","node.js","react","vue","angular","next.js","jquery"],
 ["node.js","express","nestjs","rest api","graphql","microservices"],
 ["sql","postgresql","mysql","mongodb","redis","sql server","oracle"],
 ["aws","azure","gcp","docker","kubernetes","terraform","ci/cd"],
 ["qa","testing","playwright","cypress","selenium","jest","unit testing"],
 ["python","django","flask","fastapi","pandas"],["php","laravel","symfony"],
 [".net","c#","entity framework","wcf","wpf","linq","visual studio"],
 ["java","spring","hibernate","jpa","maven","kotlin"],["html","css","bootstrap","jquery","javascript"]]
EXTRA = ["java","python","php","go","ruby","scala","kotlin","swift","c","sql","mysql","oracle","redis",
 "docker","linux","windows","jquery","bootstrap","hibernate","jpa","maven","gradle","jenkins","jira",
 "graphql","microservices","express","django","flask","entity framework","wcf","wpf","linq","visual studio",
 "azure","terraform","selenium","jest","testing","design patterns","data structures","algorithms",
 "web api","api","soap","xml","json","tdd","devops","nosql","kafka","rabbitmq","elasticsearch","spark"]
CRITICAL_MARKERS = ["must have","must-have","required","requirements","mandatory","essential","minimum","qualifications"]
OPTIONAL_MARKERS = ["nice to have","nice-to-have","preferred","plus","bonus","desired","advantage"]

CANON = {}
for g in SYNONYMS:
    for t in g: CANON[t] = g[0]
for t in EXTRA + [x for g in RELATED for x in g]:
    CANON.setdefault(t, t)
TERMS = sorted(CANON, key=len, reverse=True)
REL_OF = {}
for g in RELATED:
    for t in g: REL_OF.setdefault(CANON.get(t, t), set()).update(CANON.get(x, x) for x in g)

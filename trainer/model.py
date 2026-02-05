========== PAGE 1 ==========
Lost in Plot: Contrastive Learning for Tip-of-the-Tongue Movie Retrieval
ObedJunias
DepartmentofComputerScience
UniversityofColoradoBoulder
Boulder,CO,USA
obed.junias@colorado.edu
Abstract Totacklethisgap,weinvestigatewhethercon-
trastivelearning(Izacardetal.,2022)can enable
Humansoftenrecallamoviebyfragmentsusu-
bettersemanticretrievalofmoviesfromvagueuser
allyusingbitsofplot,emotion,orstrikingvisu-
descriptions.
alsratherthanitsexacttitle. Traditionalsearch
So,weframeourresearcharoundthefollowing
enginesandkeyword-basedretrieversstruggle
questions:
withsuchvaguequeries,whilegenerativemod-
elscanhallucinateplausiblebutincorrecttitles.
1. How effective is contrastive learning-based
Inthiswork,weproposeadenseretrievalap-
proach that leverages contrastive learning to dense retrieval for retrieving movies from
alignuserdescriptionsandmoviemetadatain vagueuserdescriptions?
asharedsemanticspace. Weevaluatethissys-
temonapublicdatasetderivedfromTMDband 2. How does this fine-tuned model compare to
IMDbandcompareitwithtwostrongbaselines: few-shot prompting methods using GPT-4
few-shotpromptingoflargelanguagemodels (OpenAI,2023)andbaseencodermodelwith-
andthevanillaencoderbeforecontrastivefine- outanyfine-tuning?
tuning. Ourexperimentsdemonstratethatthe
contrastivelyfine-tunedmodeloutperformsthe Toaddressthese:
GPT-4 few-shot baseline on both Recall@K
andMRR,butfallsshortoverthebaseencoder. 1. Webuildadenseretrievalsystemthatpullsto-
gethersemanticallysimilarmoviesandpushes
1 Introduction
apart dissimilar ones via a contrastive loss
basedonlatentthemes.
Theabilitytoidentifyaspecificmoviefromvague
natural language descriptions represents an inter-
2. We create a synthetic evaluation dataset of
section of Natural Language Processing (NLP),
3000 vague descriptions using GPT-4 few-
Information Retrieval (IR), and semantic under-
shotprompting.
standing. We often struggle to recall the name
of a movie we once saw. Instead of exact titles, 3. Weindex100kmoviesinFAISSandcompare
we remember fleeting details: a character’s tear- retrievalperformanceagainstGPT-4few-shot
ful goodbye, a tragic love story with time travel, and vanilla BERT using Recall@1/5/10/25
and this is closely related to the phenomenon of andMRR.
"tip-of-the-tongue"retrieval (Arguelloetal.,2021)
Our results show clear gains over GPT-4 few-
.
shotprompting,butfailtoevenmatchthevanilla
In practice there’s a big gap between how we
BERTencoderonRecall@KandMRR.
naturally describe a movie and how it’s stored in
adatabase. Weremembermoviesas“thatcomic-
RelatedWork
horroraboutahauntedhouseatChristmas,”or“the
onewhereBradPittplaysDeath,”notasneatkey- Tip-of-the-tongueretrievaltasksandtheirconnec-
words. These vague descriptions leave keyword- tiontoNLUhavereceivedgrowingattentiondue
searchembeddingsoftenstruggling. Thisproblem totheirreal-worldapplications. Arguelloetal.(Ar-
isn’tjustlimitedtomovies: peopledescribesongs guelloetal.,2021)intheirstudy,highlighttheneed
as “that track about flying and loss” or photos as forqueryunderstandingsystemsthatcaninterpret
“thewaterfallatsunrise.” incompletequeriesefficiently.
1
========== PAGE 2 ==========
Linetal.(Linetal.,2023)introducedamethod andmoviemetadatainasharedsemanticspace.
todecomposecomplexuserqueriesfortip-of-the-
2.1 DataPreparation
tongueretrieval. Theirapproachhighlightsthechal-
lengesinhandlinglong,semanticallyrichqueries, WestartbycombiningtheTMDb/IMDbmetadata
anaspectweaddressusingdenseretrievalmethods (plot, titles, keywords). We encode genres as a
andcontrastiveembeddings. multi-hot vector and map each release year to its
Izacardetal.(Izacardetal.,2022)proposedan decade. To get a latent “theme” signal, we run
unsupervisedcontrastivelearningapproach(Con- BERTopicoverallplotsummariesandtreateach
triever) for dense information retrieval. While movie’s topic ID as its theme label. These three
their method shows strong performance in zero- signalsguideboththetripletsamplingandtheaux-
shotretrieval,itisdomain-agnostic. Ourapproach iliaryclassificationheads.
adapts contrastive learning to the movie domain
2.2 ModelArchitecture
for enhanced semantic embeddings along with
Weuseapre-trainedBERTencoder(Devlinetal.,
knowledge-graph-baseddisambiguation.
2019)toobtaincontextualizedembeddings,apply
Zhang et al. (Zhang et al., 2022) propose La-
meanpoolingtoproduceafixed-lengthvector,and
Con,asupervisedcontrastiveframeworkthatuses
then project it into a lower-dimensional retrieval
classlabelstominebothhardpositivesandnega-
space. Concretely,foraninputtextsequencetext
tives. LaConintegratessmoothlywithpre-trained
withattentionmaskm,wethencompute
transformersandyieldsupto4.1%improvement
on GLUE and 9.4% on FewGLUE benchmarks. h = MPool (cid:0) BERT(text), m (cid:1) ∈ RH,
Thisworkdemonstratesleveraginglabelsemantics
directly in contrastive objectives, which inspired where MPool denotes element-wise mean pool-
metousetheme,genre,anddecadelabelstosam- ingoverthenon-paddedtokens. Wethenapplya
ple positives and guide auxiliary classification in learned linear projection and ℓ 2 -normalization to
movieretrieval.
getaresultantdensevectorzˆwhichservesasthe
retrievalembedding. Inaddition, weattachthree
WeuseBERTopic(Grootendorst,2022)tofind
classificationheadsforgenre,decade,andtheme
latent“themes”frommovieoverviews. BERTopic
classification,eachimplementedasasinglelinear
first embeds documents with pre-trained BERT,
layeronzˆ.
thenappliesaclass-basedTF–IDFproceduretosur-
face coherent topic clusters. In our current work,
2.3 Anchor-PositiveSampling
this provides a lightweight, unsupervised signal
To teach the model to pull semantically similar
thatweintegrateasathirdcontrastiveheadalong-
moviescloserandpushdissimilaronesapart,we
sidegenreanddecade.
samplethetrainingdatainto(a andp )where:
To assess retrieval quality, we use Recall@K i i
andMeanReciprocalRank(MRR)(Manningetal., • a isthe“anchor”: amovie’scombinedmeta-
i
2008). Recall@Kmeasuresthefractionofqueries data(plot,titles,keywords)sampledbelong-
whosecorrectmovieappearsinthetop-Kresults, ingtoaparticulartheme.
while MRR averages the inverse rank of the first
• p isa“positive”example: adifferentmovie
correcthit. Thesemetricsarewidelyusedinclosed- i
sharingthesameBERTopicthemeasthean-
set retrieval evaluations and offer a clear view
chor.
of both coverage (Recall) and ranking precision
(MRR). All the remaining examples in the batch serve as
implicitnegativesforanchora inourretrievalloss.
i
2 Methodology
2.4 Multi-TaskLoss
The main aim is to learn a joint embedding
Wecombineacontrastiveretrievallosswiththree
space where these vague user descriptions and auxiliaryclassificationlosses. Denotedbyzˆa,zˆp ∈
i i
moviemetadataareclosetogether,sothatcosine- RD the embeddings of anchor and positive i, we
similarity ranking retrieves the correct film from
definetheInfoNCEretrievallossas:
anytypeofnaturallanguagequeries.
Inthissection,wedescribehowwetrainadense L = − 1 (cid:88) N log exp (cid:0) zˆa i ·zˆp i /τ (cid:1)
retrievalmodelthatalignsvagueuserdescriptions retr N (cid:80)N exp (cid:0) zˆa·zˆp/τ (cid:1)
i=1 j=1 i j
2
========== PAGE 3 ==========
whereτ isatemperaturehyperparameter. Foreach Weholdout1,500queriesforvalidationandreport
anchor,wealsopredict: finalmetricsontheremaining1,500queries.
• A multi-label genre vector via binary cross- 3.2 Baselines
entropy.
Wecomparethreeretrievalmethods:
• Adecadelabelusingcross-entropy.
1. GPT-4few-shotprompting: Giventwoex-
• Athemelabelusingcross-entropy. amplepairs,wepromptGPT-4togenerate25
movietitlesforeachquery.
Let L , L , and L denote these
genre year theme
classification losses. We weight and sum 2. Vanillaencoder: Pre-trainedBERT-baseen-
them to form the overall training objective: codes both queries and all 100 k movies;
L= wretrLretr + wgenreLgenre + wyearLyear + wthemeLtheme. movieembeddingsgointoFAISSindex,and
weretrievebycosinesimilarity.
2.5 TrainingDetails
Wefine-tunetheentiremodel(encoder,projection, 3. Contrastive model: Similarly, fine-tuned
andheads)end-to-endusingAdamWwithweight BERT-base with projection and multi-task
decay. Keyhyperparametersinclude: headsencodesbothqueriesandmoviesmeta-
data;retrievalisagainviaFAISSoveritsnor-
• Batchsize: 16
malizedembeddings.
• Learningrate: 2×10−5 forallparameters
3.3 EvaluationMetrics
• Scheduler: Cosinedecaywith10%warm-up Wemeasuretheretrievalqualityusing:
• Epochs: 8(earlystoppingonMRR) • Recall@K:fractionofqueriesforwhichthe
correctmovieappearsinthetopK results.
• Evaluation: Encode all 100K movies into
FAISSindex,measureRecall@1/5/10/25and
• MeanReciprocalRank(MRR):
MRRonheld-outsyntheticqueries
Q
3 Experiments 1 (cid:88) 1
MRR = ,
Q rank
i
We evaluate our proposed contrastive retrieval i=1
modelonalarge-scalemoviecorpusandcompare
where rank is the position of the correct
i
it against two strong baselines. This section de-
movieforqueryi.
scribes the dataset, baselines, metrics, inference
details,andtheresultsofourexperiments. 3.4 InferenceDetails
• Indexing: We encode all 100K movies into
3.1 Dataset
a FAISS IndexFlatIP index using 512-dim
Weuseapublicmoviemetadatacollectiondrawn
normalizedembeddings.
fromTMDbandIMDb1 andthenderivethreekey
resources: • Querying: Eachqueryembeddingismatched
againsttheindextoretrievetop-25candidates.
• Corpus: A collection of 100,000+ movies,
each represented by its title, plot overview,
3.5 Results
genre,releaseyear,andtagline.
Table 1 summarizes the main findings. The con-
• Latentthemes: BERTopicisrunonallthe trastivelyfine-tunedmodeloutperformsGPT-4by
plotoverviewstoproducelatentthemes. alargemarginbutstilllagsbehindthevanillaen-
coder,indicatingroomtoimprovethearchitecture.
• Synthetic queries: We then use GPT-4 to
Fromtheabovetable1,wemakethreekeyob-
generate 3000 “vague descriptions” via the
servations:
few-shot prompting technique (e.g. “a sci-fi
romanceabouttimeloops”). • VanillaBERThandlesvagueinputssurpris-
ingly well, with a recall@1 of 13.93 % and
1https://www.kaggle.com/datasets/alanvourch/
tmdb-movies-daily-updates MRRof0.1883.
3
========== PAGE 4 ==========
Model R@1 R@5 R@10 R@25 MRR • Themegranularityiscoarse: Moviesshar-
VanillaBERT 0.1393 0.2447 0.2933 0.3713 0.1883 ingthesamehigh-leveltopicmaystillbese-
ContrastiveFine-TunedBERT 0.1140 0.1960 0.2427 0.3107 0.1540
GPT-4Few-ShotPrompting 0.0647 0.0827 0.0893 0.0913 0.0724 manticallydistant(e.g.,two“sci-fi”filmswith
verydifferentplots).
Table1: Retrievalmetricson1500syntheticqueries.
• Positivesamplingisweak: Byrelyingonly
onsharedthemes,weignorestrongersignals
• Contrastivefine-tuningyieldsaslightdrop
likeoverlappingkeywords,cast,ordirectors.
(recall@1=11.40%,MRR=0.1540). This
suggeststhatourcurrentpositivesandmulti- • Competingclassificationheads: Thegenre,
taskweightingneedfurthertuningtooutper- decadeandthemeheadscanpullthemodelin
formthebaseencoder. differentdirections. Balancingmultiplelosses
andthreeauxiliaryobjectivesisdelicate,and
• GPT-4few-shotpromptingperformsworst the classification tasks may overwhelm the
(recall@1 = 0.0647 %, MRR = 0.0724), coreretrievalsignal.
confirming that even an under-performing
domain-specificdenseretrieverbeatsthegen- 5 Conclusion
eralstate-of-the-artLLMonthistask.
Inthiswork,wehavepresentedacontrastivelearn-
ing framework for retrieving movies from vague,
Insummary,vanillaBERTsetsastrongbaseline,
“tip-of-the-tongue” descriptions. Starting from a
andGPT-4’spredictionsgeneratedusingfew-shot
pre-trainedBERTencoder,weaddedlightweight
promptingstrugglesonsparse,vaguequeries. Our
projection and classification heads. And we fine-
contrastivelyfine-tunedmodelnarrowsthegapto
tuned the model with a contrastive + auxiliary
vanillaBERTbutstillrequiresimprovements(e.g.
loss. InevaluationagainstGPT-4few-shotprompt-
hardernegatives,adjustedlossweights)toconsis-
ingandthevanillaBERTencoder,ourfine-tuned
tentlyoutpeformit.
model clearly beats the GPT-4 few-shot baseline
onRecall@KandMRR,butstillfallsshortofthe
4 Discussion
vanillaBERTencoder’sperformance.
Inthiswork,wesetouttobridgethegapbetween Despitethesegains,thereisstillalotofroomleft
how users describe a movie and how movies are toimprove. Futureworkwillexplorehardnegative
represented in an encoder’s embedding space us- mining, partial fine-tuning of only the top trans-
ing contrastive learning. Our contrastively tuned formerlayers,andevaluationonrealuserqueries
modelyieldscleargainsoverGPT-4’spredictions rather than synthetic descriptions. Incorporating
(Table 1). However, it still falls slightly behind richermetadatasignals(cast,directors)intothecon-
thevanillaencoderonRecall@KandMRR.This trastiveobjectivealsopromisestostrengthenper-
mixedresultpointstobothpromiseandchallenges formance.
incontrastivelearningformovieretrieval. Overall,thisstudydemonstratesthatevenmod-
estcontrastivefine-tuningofavanillaencodercan
Why Contrastive Fine-Tuning outperforms yieldpracticalbenefitsinareliablemovieretrieval,
GPT-4Few-Shot Theresultsclearlyshowthat makingwayformorerobustsearcherswhenexact
GPT-4 often hallucinates plausible titles instead keywordsfail.
ofgroundingitsanswers. However,alightlyfine-
tuneddenseencoderusescosinesimilarityonpre- 6 Limitations
computedembeddings,makingitmuchmorereli-
Althoughthisstudyoffersaclearproofofconcept,
ableforretrievingthecorrectfilm.
thereareafewlimitationstokeepinmind:
WhyItLagsBehindVanillaBERT Webelieve
1. Syntheticqueries. OurevaluationusesGPT-
thedropinperformancepointstoafewissues:
4-generateddescriptions,whichmaynotfully
reflect real user phrasing or distribution of
• Verylargethemespace: Alargenumberof
“tip-of-the-tongue”queries.
themes can introduce noisy or unrepresenta-
tivepositives,assomethemesmightonlyhave 2. No hard negatives. We did not incorporate
ahandfulofmovies. hardnegativemining,whichcansharplyboost
4
========== PAGE 5 ==========
contrastiveretrievalasperpriorwork. GautierIzacard,MathildeCaron,LucasHosseini,Sebas-
tianRiedel,PiotrBojanowski,ArmandJoulin,and
3. Full-model fine-tuning. We fine-tuned all EdouardGrave.2022. Unsuperviseddenseinforma-
tionretrievalwithcontrastivelearning. Transactions
BERT layers uniformly. Partial fine-tuning
onMachineLearningResearch.
(e.g., only the top 2–4 transformer blocks)
might preserve more of the base encoder’s KevinLin,KyleLo,JosephGonzalez,andDanKlein.
2023. Decomposingcomplexqueriesfortip-of-the-
generic semantics while adapting to movie
tongueretrieval. InFindingsoftheAssociationfor
retrieval.
Computational Linguistics: EMNLP 2023, pages
5521–5533, Singapore. Association for Computa-
4. Computeconstraints. Memorylimits(P100 tionalLinguistics.
with16GBVRAM)forcedsmallbatchsizes
ChristopherD.Manning,PrabhakarRaghavan,andHin-
anddownsampledtraining,potentiallyunder-
richSchütze.2008. IntroductiontoInformationRe-
utilizingthecontrastivesignal.
trieval. Cambridge University Press, Cambridge,
UK.
5. Singledataset. Resultsarereportedonpub-
licly available TMDb/IMDb split. General- OpenAI. 2023. Gpt-4 technical report. https://
openai.com/research/gpt-4.
izationtootherlanguages,genres,orclosed-
captiondescriptionsremainuntested. Zhenyu Zhang, Yuming Zhao, Meng Chen, and Xi-
aodongHe.2022. Labelanchoredcontrastivelearn-
Despite these limitations, our approach offers a ingforlanguageunderstanding. InProceedingsof
the2022ConferenceoftheNorthAmericanChap-
lightweight and scalable path to improve dense
teroftheAssociationforComputationalLinguistics:
movie retrieval, and the insights gathered above
HumanLanguageTechnologies,pages1437–1449,
willhelpusbuildstrongermodelsinthefuture. Seattle,UnitedStates.AssociationforComputational
Linguistics.
7 EthicalConsiderations
Thisworkreliesonpubliclyavailablemoviemeta-
datafromTMDbandIMDb. Whilethedatasetis
large and diverse, it might still reflect some his-
toricalbiaseswhichcouldaffectretrievalfairness.
Additionally,weuseGPT-4togenerateevaluation
queries;thesesyntheticdescriptionsmaynotcap-
turethefullvariabilityofrealuserqueries.
References
JaimeArguello,AdamFerguson,EmeryFine,Bhaskar
Mitra, Hamed Zamani, and Fernando Diaz. 2021.
Tipofthetongueknown-itemretrieval: Acasestudy
inmovieidentification. InProceedingsofthe2021
ConferenceonHumanInformationInteractionand
Retrieval, CHIIR ’21, page 5–14, New York, NY,
USA.AssociationforComputingMachinery.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deepbidirectionaltransformersforlanguageunder-
standing. InProceedingsofthe2019Conferenceof
theNorthAmericanChapteroftheAssociationfor
ComputationalLinguistics: HumanLanguageTech-
nologies,Volume1(LongandShortPapers),pages
4171–4186,Minneapolis,Minnesota.Associationfor
ComputationalLinguistics.
Maarten Grootendorst. 2022. Bertopic: Neural
topicmodelingwithaclass-basedtf-idfprocedure.
Preprint,arXiv:2203.05794.
5
# 第二节——数据思维: 理解与批判计算社会科学的经验基础

计算社会科学（Computational Social Science, CSS）的兴起，在很大程度上得益于数字时代产生的海量数据以及处理这些数据的计算能力的飞跃 (Lazer et al., 2009; Cioffi-Revilla, 2010)。数据，尤其是那些由人类互动、社会系统运行过程中自然产生的“数字足迹”（digital traces），构成了这门新兴学科进行经验探究、检验理论假设、发现社会模式乃至预测未来趋势的基础。然而，将数据视为通向社会现实的透明窗口，或仅仅当作算法输入的“原材料”，是一种危险的认识论简化 (boyd & Crawford, 2012; Kitchin, 2014)。数据并非纯粹客观、价值中立的实体；它们是复杂的社会文化与技术过程的产物，携带着其生成环境的印记、偏见与局限性。因此，培养一种深刻、审慎且批判性的“数据思维”（Data Thinking）——一种超越技术操作层面，深入探究数据认识论属性、理解其作为证据的潜能与边界、并系统性反思其认知局限的能力——对于任何希望进行有效且负责任的计算社会科学研究者而言，都是不可或缺的智识准备。本节旨在系统性地梳理与数据相关的核心认识论议题，为后续的方法论选择和操作化实践奠定坚实的基础。我们将依次探讨数据的认识论属性、数据作为社会过程知识来源的推断桥梁及其挑战、社会数据的系统性认知局限，并反思数据规模带来的认识论意涵，最终导向一种批判性的数据认知实践。

## 数据的认识论属性：作为现实表征的本质与起源

从认识论的视角审视，数据最根本的属性在于其**表征性 (Representational Nature)**。数据并非社会现象本身，而是关于这些现象在特定条件下被观察、测量和记录下来的符号化或编码化形式 (Floridi, 2010; Borgman, 2015)。这意味着数据与它所意图指向的现实之间，永远存在着一种本体论上的区分和认识论上的距离。理解这一点至关重要，因为它提醒我们，对数据的分析本质上是对现实表征的分析，而非对现实本身的直接把握。这种表征过程充满了各种**中介性 (Mediation)**。无论是传统的调查问卷、实验仪器，还是现代的传感器、网络爬虫、平台日志系统，数据的产生总是依赖于特定的工具、技术或流程 (Latour, 1987; Ihde, 1990)。这些中介物并非被动地“传递”信息，而是主动地塑造了数据的形式、内容、粒度和可分析性。例如，一个社交平台的用户界面设计、其数据收集协议、API 的开放程度，都会深刻影响研究者最终能获得什么样的数据，以及这些数据能够揭示（或遮蔽）哪些社会互动面向 (van Dijck, 2013)。因此，数据的“客观性”并非天然存在，而是由这些中介过程建构出来的，并且这种建构过程往往嵌入了设计者、技术系统乃至更广泛社会结构的预设与价值。

进一步地，数据的产生还体现了显著的**选择性 (Selectivity)**。面对无限复杂的社会现实，任何数据收集行为本质上都是一种有目的的简化和抽象 (Kitchin, 2014a)。研究者或数据生产者总是在特定理论框架、研究问题或实践需求的引导下，选择性地关注现实的某些方面，而忽略其他方面。数字时代广泛存在的“**数据化**”（Datafication）趋势——即将社会生活的各个方面（如情绪、关系、位置、偏好）转化为可量化、可存储、可分析的数据的过程 (Mayer-Schönberger & Cukier, 2013; Couldry & Mejias, 2019)——本身就是一种深刻的认识论实践。它不仅改变了我们认识世界的方式，也可能改变世界本身（后文将详述）。这种转化过程并非中立，它往往遵循特定逻辑（如商业逻辑、治理逻辑），使得某些现象更容易被“数据化”，而另一些则被边缘化或扭曲。

因此，理解数据的**生成语境 (Context of Generation)** 和**意向性 (Intentionality)** 成为批判性数据思维的关键一环。我们需要追问：这份数据最初是为什么目的被收集和创建的？是用于商业运营优化、用户监控、行政管理、科学研究，还是其他？其“**原生目的**”（Primary Purpose）往往决定了数据的内在结构、覆盖范围、时间频率、变量定义以及潜在的偏向 (Groves, 2011; Kitchin, 2014)。例如，为广告投放而收集的用户行为数据，可能在反映用户真实兴趣或社会互动方面存在系统性偏差。此外，数据的产生往往涉及**多重主体意图**的交织，包括数据产生者（用户）、数据收集者（平台/机构）、数据处理者和数据使用者（研究者）。各方可能拥有不同甚至冲突的目标和利益，这使得数据的意义解释更加复杂，需要研究者具备高度的语境敏感性。

数据的**本体论状态 (Data Ontology)** 也直接影响其**认知潜能 (Epistemic Potential)**。不同类型的数据——如图数据库（graph data）强调关系，地理空间数据（geospatial data）强调位置，文本数据（text data）蕴含意义，时间序列数据（time-series data）捕捉动态——为我们认识社会现象提供了不同的维度和切入点 (Cioffi-Revilla, 2014; Borgman, 2015)。理解特定数据类型的本体论预设（例如，网络数据通常预设了离散节点和边的存在）及其表征能力的边界至关重要。与此同时，**元数据 (Metadata)**——即关于数据的数据，描述了数据的来源、定义、格式、质量、收集方法、时间范围等信息——对于评估数据的认识论价值起着决定性作用 (Gitelman, 2013; Borgman, 2015)。缺乏充分元数据的数据往往如同失去了“身份”和“履历”，其可信度和可用性大打折扣。数据的格式和结构也赋予其不同的“**可操作性**”（Affordances）(Gibson, 1979; Kitchin, 2014)，即数据“允许”或“邀请”我们对其进行何种类型的计算分析。这种可操作性反过来又可能塑造我们的研究问题和分析路径，形成一种技术与认知的互动循环。

最后，在认识论层面厘清**观察 (Observation)**、**测量 (Measurement)** 与**数据 (Data)** 之间的概念链条极为重要。观察是对现象的初步感知，可能带有主观性和非系统性。测量则是依据特定规则，将观察到的现象属性赋予符号（通常是量化的数值）的过程 (Stevens, 1946; Michell, 1999)。测量蕴含着一系列理论和操作化假设，例如，是否可以将被测量的属性视为连续的、可加的，以及所使用的测量工具是否有效、可靠。数据则是测量结果经过系统性记录、组织和存储的产物 (Suppes & Zanotti, 1981)。计算社会科学研究所依赖的“数据”，实际上是这个从观察到测量的链条末端的输出。因此，对其认识论地位的批判性评估，必须回溯性地审视整个链条中可能引入的假设、偏差和失真 (Blalock, 1982)。例如，在线社交网络中的“连接”数据，是基于平台对“好友关系”或“关注关系”的操作化定义（测量规则）而产生的记录（数据），它与个体实际的社会支持网络（观察到的现象或潜在构念）之间可能存在显著差异。忽视这一链条，直接将数据等同于现实，是认识论上的重大谬误。

## 数据作为社会过程的知识来源：推断的桥梁与挑战

数据之所以在计算社会科学中扮演核心角色，是因为它被视为理解和解释社会过程——如人类行为模式、社会结构演化、文化意义传播、集体动态涌现——的**经验证据 (Empirical Evidence)** (Popper, 1959; Hempel, 1965)。然而，数据本身并不能自动生成知识。从数据记录到关于社会过程的知识主张，必须跨越一个**推断的桥梁 (Inferential Bridge)**。这个推断过程本质上是基于归纳、演绎或溯因逻辑，并且高度依赖于研究者所持有的理论假设、背景知识、分析框架以及所采用的计算方法 (Pearl, 2009; Salganik, 2018)。数据只是推断链条中的一环，其作为证据的强度（informativeness and relevance）取决于它与所要探究的社会现象的关联程度、数据本身的质量，以及研究者在多大程度上能够排除其他可能的解释或混淆因素 (Campbell & Stanley, 1963)。

社会数据能够为我们提供关于社会过程不同层面的洞见。例如，用户的点击流、购买记录、地理移动轨迹等可以被视为**行为痕迹 (Behavioral Traces)**，揭示个体和群体的日常实践、决策模式与互动节奏 (Lazer et al., 2009)。社交网络图、组织成员名单、合著网络等数据则像是**结构快照 (Structural Snapshots)**，表征了特定时间点上的社会关系、权力分布或合作模式 (Wasserman & Faust, 1994; Borgatti et al., 2013)。大量的文本、图像、视频等多模态数据则成为**意义载体 (Carriers of Meaning)**，承载着个体的观点、情感、文化符号、社会规范和意识形态 (Jurafsky & Martin, 2023)。而带有时间戳的事件数据、面板数据（longitudinal data）或连续生成的流数据（streaming data）则记录了**动态流变 (Dynamic Flows)**，使研究者得以追踪社会现象的演变轨迹、扩散过程和因果时序 (Coleman, 1981; Cioffi-Revilla, 2014)。然而，对于每一种数据表征，研究者都需要审慎思考其局限性。行为痕迹可能无法反映行为背后的动机和意图；结构快照可能是静态的，忽略了关系的动态变化；意义的解读高度依赖于阐释框架和文化语境；动态记录往往是离散的，可能无法捕捉到关键的转折点或连续过程的细微变化。

这种对数据意义解读的**语境依赖性 (Context Dependency)** 是认识论上的一个核心挑战。数据的含义并非内在于符号本身，而是在其产生的具体社会、文化、历史背景以及研究者所采用的理论视角下被赋予和建构的 (Geertz, 1973; Gitelman, 2013)。例如，社交媒体上的一个“点赞”行为，在不同平台、不同文化、不同用户关系、不同内容语境下，可能代表着赞同、已阅、社交礼仪、甚至讽刺等截然不同的含义。脱离语境的数据聚合和量化分析，很可能导致对社会现象的肤浅理解甚至严重误读。因此，计算社会科学研究不仅需要强大的计算能力，同样需要深厚的领域知识（domain knowledge）和对社会文化语境的敏感性 (Schrodt, 2014)。

更进一步，我们需要认识到数据与现实之间可能存在的**互动关系 (Interaction between Data and Reality)**，这超越了数据仅仅作为现实被动“记录”的传统观念。一方面，数据化过程本身就可能**构成 (Constitute)** 或**塑造 (Shape)** 它所意图测量的社会现实。正如伊恩·哈金（Ian Hacking）所提出的“循环效应”（looping effect）概念，对人类行为的分类和测量会反过来影响人们的自我认知和行为方式 (Hacking, 1995)。在数字时代，这种效应尤为显著。例如，在线声誉系统、信用评分、健康追踪应用等，不仅仅是记录用户的状态，它们通过设定标准、提供反馈、影响机会，直接干预和塑造着用户的行为与身份认同 (Zuboff, 2019)。这种数据的“**展演性**”（Performativity）——即数据不仅描述世界，也参与构建世界——使得认识论上的因果推断和客观性声称变得异常复杂 (MacKenzie, 2006)。研究者必须警惕，他们分析的数据可能并非“自然”状态下的社会现象，而是已经被数据化过程本身所“污染”或“改造”的结果。这种数据与现实之间的反馈循环，对计算社会科学研究的设计、解释和伦理提出了深刻的挑战。

## 社会数据的系统性认知局限：知识扭曲的来源分类

尽管数据为计算社会科学提供了前所未有的经验基础，但我们必须清醒地认识到，所有数据，尤其是大规模、被动收集的数字痕迹数据，都内嵌着各种系统性的局限，这些局限可能扭曲我们对社会现实的认知，损害知识主张的有效性和可靠性。批判性的数据思维要求我们能够系统性地识别、评估这些局限的来源及其潜在的认识论后果。我们可以将这些局限大致归纳为以下几个相互关联的类别：

表 1：社会数据的系统性认知局限来源分类及其认识论后果

| 局限类别     | 主要表现                                                  | 核心认识论问题      | 潜在后果                                             |
| -------- | ----------------------------------------------------- | ------------ | ------------------------------------------------ |
| 来源与覆盖局限  | <p>人口/群体代表性偏差<br>内容/行为覆盖偏差<br>时间/空间覆盖偏差</p>           | 我们看到了谁/什么？   | <p>抽样偏差<br>结论的外部效度受损<br>对特定群体或现象的系统性忽视</p>       |
| 测量与表征局限  | <p>构念-测量差距<br>测量误差 (随机/系统)<br>代理变量问题<br>观察者效应/反应性</p> | 我们看到的是否准确？   | <p>测量偏差<br>构念效度低下<br>关系估计失真<br>行为非自然</p>         |
| 系统与中介局限  | <p>算法混淆/中介<br>系统动态性/非平稳性<br>反馈循环/内生性</p>              | 我们看到的受到何种干扰？ | <p>因果推断困难<br>结果不可复现/不稳定<br>模型预测失效</p>            |
| 数据内在质量局限 | <p>数据缺失 (模式)<br>数据噪音<br>数据真实性/伪造<br>数据一致性/整合问题</p>    | 数据本身可靠吗？     | <p>信息损失<br>分析结果偏差/方差增大<br>结论基于虚假信息<br>数据融合困难</p> |

下面我们对每一类局限进行更详细的阐述：

* **来源与覆盖局限 (Limitations from Source and Coverage)**：这类局限源于数据并非对目标研究总体（无论是人群、行为、内容、时间还是空间）的全面、无偏反映。**人口/群体代表性偏差**是数字数据面临的普遍问题 (Schober et al., 2016; Hargittai, 2020)。例如，特定社交媒体平台的用户在年龄、性别、教育程度、地理分布、政治倾向等方面可能与总体人口或其他平台用户存在显著差异（所谓的“平台偏差” Platform Bias）。更深层次的**数字鸿沟 (Digital Divide)** 意味着那些缺乏数字接入或数字技能的群体在数据世界中被系统性地“消声” (van Dijk, 2020)。**内容/行为覆盖偏差**则指数据可能只捕捉了现象的某些特定方面。例如，在线购物数据反映了消费行为，但可能无法揭示人们的线下社交互动；政治讨论论坛的数据反映了言论，但可能忽略了非言语的政治参与。**时间/空间覆盖偏差**也很常见，例如，某些历史时期的数据可能难以获取，或者数据只覆盖了特定城市而无法代表更广泛区域。这些覆盖性的局限直接威胁到研究结论的**外部效度 (External Validity)**，即研究结果能否推广到数据来源之外的人群、情境或时间段 (Shadish, Cook, & Campbell, 2002)。忽视这些偏差可能导致研究者基于有偏样本得出关于普遍规律的错误结论。
* **测量与表征局限 (Limitations from Measurement and Representation)**：即使数据来源覆盖了目标群体或现象，其测量和表征方式也可能存在问题。核心挑战在于**构念-测量差距**，即研究中实际测量或操作化的变量（如“点赞数”、“好友数”、“在线时长”）与研究者真正关心的抽象理论构念（如“社会影响力”、“社会资本”、“幸福感”）之间往往存在距离 (Cronbach & Meehl, 1955; Goertz, 2006)。这种差距导致了**构念效度 (Construct Validity)** 问题——我们测量的是我们声称要测量的东西吗？许多数字痕迹数据本质上是研究者无法直接控制其测量过程的“现成数据”(found data)，这使得构念效度评估尤为困难 (Howison, Wiggins, & Crowston, 2011)。**测量误差**是所有经验研究都面临的问题，包括随机误差（噪音）和系统误差（偏差）。在数字环境中，平台界面变化、用户误操作、数据记录系统的 bug 等都可能引入测量误差。**代理变量问题**也很突出，研究者常常不得不使用容易获取但并非最优的代理指标来测量难以直接观察的构念，这可能引入额外的偏差和噪音 (Bound, Brown, & Mathiowetz, 2001)。此外，如前所述，**观察者效应或反应性**在数字环境中可能以新的形式出现，例如用户意识到自己的行为被追踪或评估时，可能会改变其行为模式（例如，为了迎合算法推荐而调整浏览习惯），使得观测到的数据并非其“自然”状态下的表现 (Acquisti, Taylor, & Wagman, 2016)。
* **系统与中介局限 (Limitations from System and Mediation)**：现代数字数据通常产生于复杂的社会技术系统之中，这些系统本身及其运行逻辑会深刻影响我们能观测到的数据，带来独特的认识论挑战。**算法混淆或中介**是计算社会科学面临的核心难题之一 (Salganik, 2018; Anderson et al., 2020)。搜索引擎的排序算法、社交媒体的信息流推荐算法、内容审核机制等，都在无形中塑造着用户接触到的信息、互动模式和行为选择。这使得研究者很难区分观测到的模式是源于用户内生的偏好或行为，还是算法干预的结果。例如，社交网络中的“同质性”（homophily，相似的人倾向于连接）可能被推荐算法放大，使得从观测数据中估计真实的同质性水平变得困难。**系统动态性与非平稳性**也带来挑战。社会系统本身在不断演变，同时，数据产生的平台、用户群体、使用行为、乃至平台自身的算法和功能也在随时间变化 (Ribes & Jackson, 2013)。这使得跨时间的数据比较和建立稳定预测模型变得困难，因为过去的模式可能不再适用于未来（即“模型漂移” model drift）。**反馈循环与内生性**问题也更为突出，例如，基于历史犯罪数据部署警力的预测性警务系统，可能导致在特定区域投入更多警力，进而发现更多犯罪，产生的数据又进一步强化了最初的偏见，形成恶性循环 (Ensign et al., 2018)。这种内生性使得从观测数据中识别真实的因果关系异常困难。
* **数据内在质量局限 (Limitations from Intrinsic Data Quality)**：最后，数据本身可能存在各种质量问题，直接影响分析结果的可靠性。**数据缺失**是常见问题，关键在于缺失的机制：是完全随机缺失(MCAR)、随机缺失(MAR)还是非随机缺失(MNAR)？不同的缺失机制对分析结果的偏差影响不同，需要采用恰当的方法处理 (Little & Rubin, 2002)。大规模数据中往往也伴随着相当程度的**数据噪音**，即无关或错误的记录。**数据真实性与伪造**问题在数字时代变得更加突出，例如社交媒体上的虚假账户（bots）、刷单行为、网络水军等产生的“脏数据”可能严重污染分析结果 (Ferrara et al., 2016)。对于依赖用户生成内容的平台尤其如此。**数据一致性与整合问题**则出现在需要合并来自不同来源、不同时间点或不同格式的数据时，可能存在定义不一致、格式不兼容、实体识别困难等问题 (Doan, Halevy, & Ives, 2012)。

这些系统性局限相互交织，共同构成了计算社会科学研究在认识论上面临的严峻挑战。其最终的**认知后果**是导致基于数据的知识主张充满了**不确定性 (Uncertainty)**。批判性的数据思维并不意味着因噎废食、放弃使用数据，而是要求研究者在整个研究过程中，始终对这些潜在的局限性保持警惕，主动地去识别、评估它们对研究结论可能产生的影响，并通过研究设计（如多源数据印证）、分析方法（如敏感性分析、偏差校正模型）和结果解释（如明确承认局限、界定结论适用范围）来尽可能地缓解或说明这些问题。这要求研究者不仅具备技术能力，更要具备深刻的认识论自觉和方法论审慎。

## 数据规模的认识论意涵：新视野、旧陷阱与批判性反思

计算社会科学常常与“大数据”(Big Data)联系在一起，后者通常被认为具有 “3V”（体量大、速度快、多样性）特征 (Laney, 2001; Kitchin, 2014)。数据规模的急剧扩张无疑为社会科学研究打开了**新的认识论视野 (New Epistemic Horizons)**。首先，大规模数据使得研究者能够**探测到传统小样本研究难以发现的微弱信号、罕见事件或细微差异** (Fan, Han, & Liu, 2014)。其次，数据的多样性和高维度允许我们**分析更为复杂的社会互动模式和网络结构**，捕捉到个体行为与宏观现象之间的多层次联系 (Contractor et al., 2006)。再次，数据的实时性或高频性使得**近乎实时地追踪社会动态过程**（如舆情演变、疾病传播、金融市场波动）成为可能 (Sakaki, Okazaki, & Matsuo, 2010)。最后，数据在时间和空间上的延展性使得研究**更大时空尺度的社会变迁和模式**成为可能 (Aiden & Michel, 2013)。这些无疑极大地拓展了社会科学的研究议程和解释能力。

然而，数据规模的扩张并非全然是认识论上的福音，它也可能放大已有的**旧陷阱 (Old Traps, Amplified)**，并带来新的认识论风险。一个突出的风险是**对虚假关联的易感性显著增加** (Calude & Longo, 2017; Harford, 2014)。当变量数量巨大、数据点极多时，仅仅因为偶然性就发现统计上显著（例如 p 值很小）但实际上毫无意义或误导性的相关关系的可能性大大增加。这要求研究者在解释相关性时更加审慎，并更加注重因果推断的严格性。另一个陷阱是所谓的“**N=All 的幻觉**” (boyd & Crawford, 2012)。即使数据规模巨大，它通常也并非真正意义上的“全数据”(total population data)，而只是“可用数据”(available data)。如前所述，这些数据往往带有系统性的覆盖偏差。规模的庞大性可能掩盖甚至加剧这些偏差的影响，使得研究者误以为自己的发现具有普遍性，而实际上可能只适用于产生数据的特定（且可能有偏的）人群或情境。此外，大规模、高维度数据也增加了**模型过度拟合 (Overfitting)** 的风险，即模型过于完美地拟合了训练数据中的噪音和特异性，导致其在新的、未见过的数据上表现很差，丧失了**泛化能力 (Generalizability)** (Hastie, Tibshirani, & Friedman, 2009)。最后，过度强调数据的规模和计算能力，可能带来**理论空洞化的风险** (Risk of A-theoretical Empiricism)。一些人甚至宣称大数据时代“理论的终结”(Anderson, 2008)，认为数据自身就能“说话”，无需理论指导。这是一种危险的认识论短视 (Pietsch, 2016; Kitchin, 2014)。缺乏理论指导的数据分析容易陷入盲目的“数据挖掘”或仅仅发现肤浅的相关性，而无法提供深刻的解释机制和对社会过程的真正理解。理论在提出有意义的研究问题、构建合理的模型、解释分析结果以及评估发现的重要性方面，仍然发挥着不可替代的作用 (Hedström & Ylikoski, 2010; Watts, 2014)。

数据规模的扩张及其带来的计算方法（如机器学习）的广泛应用，也对一些**传统的认识论假设提出了挑战 (Challenges to Traditional Epistemological Assumptions)**。例如，当数据规模接近甚至号称覆盖总体时，传统基于小样本抽样的**统计推断逻辑**（如置信区间、假设检验）的解释和适用性引发了讨论 (Lin, Lucas Jr, & Shmueli, 2013; Meng, 2018)。同时，尽管数据量巨大，但从观测性大数据中进行可靠的**因果推断**仍然面临巨大的方法论和认识论障碍（如混淆变量控制、内生性问题） (Pearl & Mackenzie, 2018; Athey, 2017)。此外，机器学习算法在预测方面的成功，特别是那些“黑箱”模型，引发了关于**知识的可解释性 (Interpretability)** 与**预测准确性 (Predictive Accuracy)** 之间关系的讨论：一个我们不完全理解其内部机制但预测准确的模型，能在多大程度上构成“科学知识”？(Rudin, 2019; Lipton, 2018)。知识发现过程的**自动化**也引发了关于“发现”主体性（是人类研究者还是算法？）以及知识**正当性 (Justification)** 来源的新问题。

面对这些机遇、陷阱和挑战，计算社会科学研究者需要发展并践行一套**批判性的数据认知实践 (Critical Data Epistemic Practices)**。这包括：

* **数据谱系追溯 (Data Provenance Tracking)**：系统性地记录、审视和报告数据的来源、生成过程、处理历史及其潜在影响 (Borgman, 2015; Chapman et al., 2020)。
* **情境化理解 (Contextual Understanding)**：将数据分析嵌入其产生的具体社会、技术和组织背景中，结合定性知识和领域专长进行解读 (boyd & Crawford, 2012)。
* **方法论透明性 (Methodological Transparency)**：清晰、完整地报告数据处理步骤、分析方法、模型假设和参数选择，确保研究过程的可检验性和可复现性 (Peng, 2011)。
* **知识主张的审慎性 (Prudence in Knowledge Claims)**：坦诚地承认研究的不确定性和局限性，避免对结果进行过度泛化或过度自信的因果声称 (Lazer et al., 2014)。
* **多源数据/方法三角互证 (Triangulation)**：尽可能结合使用不同来源、不同类型的数据（如数字痕迹、调查数据、实验数据、定性访谈）和不同的分析方法，对研究发现进行交叉验证，以增强结论的稳健性 (Jick, 1979; Webb et al., 1966; Edelmann, Wolff, & Bail, 2020)。
* **理论敏感性 (Theoretical Sensitivity)**：始终保持数据分析与社会科学理论的对话，用理论指导数据探索的方向和结果的解释，同时用数据分析中的发现来检验、修正或启发理论创新 (Timmermans & Tavory, 2012; Watts, 2014)。

批判性的数据思维还需要包含**认识论反身性 (Epistemological Reflexivity)** (Bourdieu & Wacquant, 1992; Alvesson & Sköldberg, 2009)。研究者需要意识到，数据并非仅仅是被动观测的对象，数据、算法以及基于它们产生的知识本身具有**能动性 (Agency)**，它们可以反过来干预、塑造甚至“制造”社会现实，这就是数据的“**展演性**”(performitivity) 在认识论上的深刻意涵 (MacKenzie, 2006)。例如，预测犯罪的模型可能改变警务实践，影响犯罪率的记录方式；推荐算法不仅反映用户偏好，也可能塑造用户的偏好。研究者自身作为数据分析和知识生产者，也嵌入在这个过程中，其自身的立场、价值观、研究选择都可能影响研究结果及其社会后果。因此，反思研究者自身在知识生产过程中的角色、权力关系以及研究工作可能带来的潜在社会影响，是负责任的计算社会科学实践不可或缺的一部分 (D'Ignazio & Klein, 2020)。

“数据思维”远不止于掌握数据处理和分析的技术技能。它是一种根本性的认识论实践，要求研究者对数据的本质持有批判性理解，深刻认识到数据作为社会现实表征的复杂性、中介性和局限性。从数据的起源、生成语境，到其作为证据的推断桥梁，再到系统性的认知局限（覆盖、测量、系统、质量），以及数据规模带来的双重影响，每一个环节都需要审慎的考量。计算社会科学研究者必须培养一种能力，既能充分利用数据带来的前所未有的洞察力，又能时刻警惕数据中潜藏的陷阱和偏见，并在整个研究过程中保持方法论的透明性、理论的敏感性和认识论的反身性。只有这样，我们才能在数字化的浪潮中，更加审慎、可靠且富有洞察力地推进对社会世界的理解。这种批判性的数据思维，将为我们在后续章节中探讨具体的方法论原则和操作技术奠定坚实的认识论基础，并最终导向更有意义、更有价值的计算社会科学研究。下一节将基于问题思维、数据思维，进一步地讨论模型思维。

## 参考文献

```
Acquisti, Alessandro, Curtis Taylor, and Liad Wagman. 2016. “The Economics of Privacy.” Journal of Economic Literature 54(2): 442–92. doi:10.1257/jel.54.2.442.
Aiden, Erez, and Jean-Baptiste Michel. 2013. Uncharted: Big Data as a Lens on Human Culture. First Edition. New York: Riverhead Books.
Alvesson, Mats, and Kaj Skoldberg. 2009. Reflexive Methodology: New Vistas For Qualitative Research. Second edition. Los Angeles ; London: Sage Publications Ltd.
Anderson, David, Dennis Sweeney, Thomas Williams, Jeffrey Camm, and James Cochran. 2019. Statistics for Business & Economics. 14th edition. Boston, MA: Cengage Learning.
Anderson, Terry, ed. 2008. The Theory and Practice of Online Learning, Second Edition. 2nd edition. Edmonton: Athabasca University Press.
Athey, Susan, and Guido W. Imbens. 2017. “The State of Applied Econometrics: Causality and Policy Evaluation.” Journal of Economic Perspectives 31(2): 3–32. doi:10.1257/jep.31.2.3.
Blalock, Jane W. 1982. “Persistent Auditory Language Deficits in Adults with Learning Disabilities.” Journal of Learning Disabilities 15(10): 604–9. doi:10.1177/002221948201501010.
Borgatti, Stephen P., Martin G. Everett, and Jeffrey C. Johnson. 2018. Analyzing Social Networks. 2nd edition. Los Angeles: SAGE.
Borgman, Christine L. 2015. Big Data, Little Data, No Data: Scholarship in the Networked World. The MIT Press. doi:10.7551/mitpress/9963.001.0001.
Bourdieu, Pierre, and Loïc J. D. Wacquant. 1992. An Invitation to Reflexive Sociology. First Edition. Chicago: University of Chicago Press.
Boyd, Danah, and Kate Crawford. 2012. “Critical Questions for Big Data: Provocations for a Cultural, Technological, and Scholarly Phenomenon.” Information, Communication & Society 15(5): 662–79. doi:10.1080/1369118X.2012.678878.
Calude, Cristian S., and Giuseppe Longo. 2016. “The Deluge of Spurious Correlations in Big Data.” Foundations of Science 22(3): 595–612. doi:10.1007/s10699-016-9489-4.
Campbell, Donald T., and Julian Stanley. 1963. Experimental and Quasi-Experimental Designs for Research. 1st edition. Belomt, CA: Cengage Learning.
Chapman, Evelina, Michelle M. Haby, Tereza Setsuko Toma, Maritsa Carla de Bortoli, Eduardo Illanes, Maria Jose Oliveros, and Jorge O. Maia Barreto. 2020. “Knowledge Translation Strategies for Dissemination with a Focus on Healthcare Recipients: An Overview of Systematic Reviews.” Implementation Science 15(1): 14. doi:10.1186/s13012-020-0974-3.
Cioffi-Revilla, Claudio. 2010. “Computational Social Science.” WIREs Computational Statistics 2(3): 259–71. doi:10.1002/wics.95.
Cioffi-Revilla, Claudio. 2017. “Social Networks.” In Introduction to Computational Social Science: Principles and Applications, ed. Claudio Cioffi-Revilla. Cham: Springer International Publishing, 141–92. doi:10.1007/978-3-319-50131-4_4.
Coleman, Eli. 1981. “Counseling Adolescent Males.” The Personnel and Guidance Journal 60(4): 215–18. doi:10.1002/j.2164-4918.1981.tb00284.x.
Contractor, Noshir S., Stanley Wasserman, and Katherine Faust. 2006. “Testing Multitheoretical, Multilevel Hypotheses About Organizational Networks: An Analytic Framework and Empirical Example.” Academy of Management Review 31(3): 681–703. doi:10.5465/amr.2006.21318925.
Couldry, Nick, and Ulises A. Mejias. 2019. “Data Colonialism: Rethinking Big Data’s Relation to the Contemporary Subject.” Television & New Media 20(4): 336–49. doi:10.1177/1527476418796632.
Cronbach, Lee J., and Paul E. Meehl. 1955. “Construct Validity in Psychological Tests.” Psychological Bulletin 52(4): 281–302. doi:10.1037/h0040957.
D’Ignazio, Catherine, and Lauren F. Klein. 2020. Data Feminism. Cambridge: The MIT Press.
van Dijck, José. 2013. “Engineering Sociality in a Culture of Connectivity.” In The Culture of Connectivity: A Critical History of Social Media, ed. Jose van Dijck. Oxford University Press, 0. doi:10.1093/acprof:oso/9780199970773.003.0001.
Dijk, Jan van. 2019. The Digital Divide. 1st edition. Polity.
Doan, AnHai, Alon Halevy, and Zachary Ives. 2012. Principles of Data Integration. 1st edition. Waltham, MA: Morgan Kaufmann.
Edelmann, Achim, Tom Wolff, Danielle Montagne, and Christopher A. Bail. 2020. “Computational Social Science and Sociology.” Annual Review of Sociology 46(1): 61–81. doi:10.1146/annurev-soc-121919-054621.
Ensign, Danielle, Sorelle A. Friedler, Scott Neville, Carlos Scheidegger, and Suresh Venkatasubramanian. 2018. “Runaway Feedback Loops in Predictive Policing.” In Proceedings of the 1st Conference on Fairness, Accountability and Transparency, PMLR, 160–71. https://proceedings.mlr.press/v81/ensign18a.html (May 8, 2025).
Fan, Jianqing, Fang Han, and Han Liu. 2014. “Challenges of Big Data Analysis.” National Science Review 1(2): 293–314. doi:10.1093/nsr/nwt032.
FerraraEmilio, VarolOnur, DavisClayton, MenczerFilippo, and FlamminiAlessandro. 2016. “The Rise of Social Bots.” Communications of the ACM 59(7): 96–104. doi:10.1145/2818717.
Floridi, Luciano. 2010. Information: A Very Short Introduction. New York: Oxford University Press.
Geertz, Clifford. 1973. Interpretation of Cultures. Fifth Pr. edition. New York, NY: Basic Books, Inc.
Gibson, James J. 1979. The Ecological Approach to Visual Perception: Classic Edition. Houghton Mifflin.
Gitelman, Lisa, ed. 2013. Raw Data Is an Oxymoron. Cambridge (Mass.): Mit Pr.
Gitelman, Lisa, Virginia Jackson, Daniel Rosenberg, Travis D. Williams, Kevin R. Brine, Mary Poovey, Matthew Stanley, et al. 2013. “Data Bite Man: The Work of Sustaining a Long-Term Study.” In “Raw Data” Is an Oxymoron, MIT Press, 147–66. https://ieeexplore.ieee.org/document/6462156 (May 9, 2025).
Goertz, Gary. 2006. Social Science Concepts: A User’s Guide. Princeton University Press. doi:10.2307/j.ctvcm4gmg.
Groves, Robert M. 2011. “Three Eras of Survey Research.” Public Opinion Quarterly 75(5): 861–71. doi:10.1093/poq/nfr057.
Hacking, Ian. 1995. Rewriting the Soul:  Multiple Personality and the Sciences of Memory. Princeton, NJ, US: Princeton University Press.
Harford, Tim. 2014. “Big Data: A Big Mistake?” Significance 11(5): 14–19. doi:10.1111/j.1740-9713.2014.00778.x.
Hargittai, Eszter. 2020. “Potential Biases in Big Data: Omitted Voices on Social Media.” Social Science Computer Review 38(1): 10–24. doi:10.1177/0894439318788322.
Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009a. “Basis Expansions and Regularization.” In The Elements of Statistical Learning: Data Mining, Inference, and Prediction, eds. Trevor Hastie, Robert Tibshirani, and Jerome Friedman. New York, NY: Springer, 139–89. doi:10.1007/978-0-387-84858-7_5.
Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009b. “Model Inference and Averaging.” In The Elements of Statistical Learning: Data Mining, Inference, and Prediction, eds. Trevor Hastie, Robert Tibshirani, and Jerome Friedman. New York, NY: Springer, 261–94. doi:10.1007/978-0-387-84858-7_8.
Hedström, Peter, and Petri Ylikoski. 2010. “Causal Mechanisms in the Social Sciences.” Annual Review of Sociology 36(Volume 36, 2010): 49–67. doi:10.1146/annurev.soc.012809.102632.
Hempel, Carl Gustav. 1965. Aspects of Scientific Explanation and Other Essays in the Philosophy of Science. New York: The Free Press.
Howison, James, Andrea Wiggins, and Kevin Crowston. 2011. “Validity Issues in the Use of Social Network Analysis with Digital Trace Data.” Journal of the Association for Information Systems 12(12). doi:10.17705/1jais.00282.
Ihde, Don. 1990. Technology and the Lifeworld: From Garden to Earth. Bloomington: Indiana University Press.
Jick, Todd D. 1979. “Mixing Qualitative and Quantitative Methods: Triangulation in Action.” Administrative Science Quarterly 24(4): 602–11. doi:10.2307/2392366.
Jurafsky, Daniel, and James Martin. 2008. Speech and Language Processing, 2nd Edition. 2nd edition. Upper Saddle River, NJ: Prentice Hall.
Kitchin, Rob. 2014. “Big Data, New Epistemologies and Paradigm Shifts.” Big Data & Society 1(1): 2053951714528481. doi:10.1177/2053951714528481.
Latour, Bruno. 1988. Science in Action: How to Follow Scientists and Engineers Through Society. Revised ed. edition. Cambridge (Mass.): Harvard University Press.
Lazer, David, Ryan Kennedy, Gary King, and Alessandro Vespignani. 2014. “Big Data. The Parable of Google Flu: Traps in Big Data Analysis.” Science (New York, N.Y.) 343(6176): 1203–5. doi:10.1126/science.1248506.
Lazer, David, Alex Pentland, Lada Adamic, Sinan Aral, Albert-László Barabási, Devon Brewer, Nicholas Christakis, et al. 2009. “Computational Social Science.” Science 323(5915): 721–23. doi:10.1126/science.1167742.
Lin, Mingfeng, Henry C. Lucas Jr., and Galit Shmueli. 2013. “Too Big to Fail: Large Samples and the p-Value Problem.” Information Systems Research 24(4): 906–17. doi:10.1287/isre.2013.0480.
Lipton, Zachary C. 2018. “The Mythos of Model Interpretability: In Machine Learning, the Concept of Interpretability Is Both Important and Slippery.” Queue 16(3): 31–57. doi:10.1145/3236386.3241340.
Little, Roderick J. A., and Donald B. Rubin. 2002a. “Large-Sample Inference Based on Maximum Likelihood Estimates.” In Statistical Analysis with Missing Data, John Wiley & Sons, Ltd, 190–99. doi:10.1002/9781119013563.ch9.
Little, Roderick J. A., and Donald B. Rubin. 2002b. “Maximum Likelihood for General Patterns of Missing Data: Introduction and Theory with Ignorable Nonresponse.” In Statistical Analysis with Missing Data, John Wiley & Sons, Ltd, 164–89. doi:10.1002/9781119013563.ch8.
Little, Roderick J. A., and Donald B. Rubin. 2002c. “Missing Data in Experiments.” In Statistical Analysis with Missing Data, John Wiley & Sons, Ltd, 24–40. doi:10.1002/9781119013563.ch2.
MacKenzie, Doris Layton. 2006. What Works in Corrections:  Reducing the Criminal Activities of Offenders and Delinquents. New York, NY, US: Cambridge University Press. doi:10.1017/CBO9780511499470.
Mayer-Schönberger, Viktor, and Kenneth Cukier. 2013. Big Data: A Revolution That Will Transform How We Live, Work, and Think. Boston, MA: Houghton Mifflin Harcourt.
“Measurement Error in Survey Data.” 2001. In Handbook of Econometrics, Elsevier, 3705–3843. doi:10.1016/S1573-4412(01)05012-7.
Meng, Xiao-Li. 2018. “Statistical Paradises and Paradoxes in Big Data (I): Law of Large Populations, Big Data Paradox, and the 2016 US Presidential Election.” The Annals of Applied Statistics 12(2): 685–726. doi:10.1214/18-AOAS1161SF.
Mf, Schober, Pasek J, Guggenheim L, Lampe C, and Conrad Fg. 2016. “Social Media Analyses for Social Measurement.” Public opinion quarterly 80(1): 180–211. doi:10.1093/poq/nfv048.
Michell, Joel. 1999. Measurement in Psychology: A Critical History of a Methodological Concept. New York, NY, US: Cambridge University Press. doi:10.1017/CBO9780511490040.
Pearl, Judea. 2009. Causality: Models, Reasoning and Inference. 2nd edition. Cambridge, U.K. ; New York: Cambridge University Press.
Pearl, Judea, and Dana Mackenzie. 2018. The Book of Why: The New Science of Cause and Effect. 1st edition. New York: Basic Books.
Peng, Roger D. 2011. “Reproducible Research in Computational Science.” Science (New York, N.Y.) 334(6060): 1226–27. doi:10.1126/science.1213847.
Popper, Karl R. 1959. The Logic of Scientific Discovery. Oxford, England: Basic Books.
Rudin, Cynthia. 2019. “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.” Nature Machine Intelligence 1(5): 206–15. doi:10.1038/s42256-019-0048-x.
Sakaki, Takeshi, Makoto Okazaki, and Yutaka Matsuo. 2010. “Earthquake Shakes Twitter Users: Real-Time Event Detection by Social Sensors.”
Salganik, Matthew J. 2018. Bit by Bit: Social Research in the Digital Age. Princeton, NJ, US: Princeton University Press.
Shadish, William R., Thomas D. Cook, and Donald T. Campbell. 2002. Experimental and Quasi-Experimental Designs for Generalized Causal Inference. Boston, MA, US: Houghton, Mifflin and Company.
Stevens, S. S. 1946. “On the Theory of Scales of Measurement.” Science 103: 677–80. doi:10.1126/science.103.2684.677.
Suppes, Patrick, and Mario Zanotti. 1981. “When Are Probabilistic Explanations Possible?” Synthese 48(2): 191–99. doi:10.1007/BF01063886.
Timmermans, Stefan, and Iddo Tavory. 2012. “Theory Construction in Qualitative Research: From Grounded Theory to Abductive Analysis.” Sociological Theory 30(3): 167–86. doi:10.1177/0735275112457914.
Wasserman, Stanley, and Katherine Faust. 1994. Social Network Analysis:  Methods and Applications. New York, NY, US: Cambridge University Press. doi:10.1017/CBO9780511815478.
Watts, Simon. 2014. “User Skills for Qualitative Analysis: Perspective, Interpretation and the Delivery of Impact.” Qualitative Research in Psychology 11(1): 1–14. doi:10.1080/14780887.2013.776156.
Webb, Eugene J., Donald T. Campbell, Richard D. Schwartz, and Lee Sechrest. 1966. Unobtrusive Measures: Nonreactive Research in the Social Sciences. Oxford, England: Rand Mcnally.
Zuboff, Shoshana. 2019. “Surveillance Capitalism and the Challenge of Collective Action.” New Labor Forum 28(1): 10–29. doi:10.1177/1095796018819461.
```

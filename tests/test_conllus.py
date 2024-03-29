text_testfile1 = '''
# sent_id = reviews-071650-0010
# text = I would understand if I was being treated this way by a staff member but the club's actual OWNER?!
1	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	3	nsubj	3:nsubj|20:nsubj	_
2	would	would	AUX	MD	VerbForm=Fin	3	aux	3:aux	_
3	understand	understand	VERB	VB	VerbForm=Inf	0	root	0:root	_
4	if	if	SCONJ	IN	_	8	mark	8:mark	_
5	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	8	nsubj:pass	8:nsubj:pass	_
6	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin	8	aux	8:aux	_
7	being	be	AUX	VBG	VerbForm=Ger	8	aux:pass	8:aux:pass	_
8	treated	treat	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	3	advcl	3:advcl:if	_
9	this	this	DET	DT	Number=Sing|PronType=Dem	10	det	10:det	_
10	way	way	NOUN	NN	Number=Sing	8	obj	8:obj	_
11	by	by	ADP	IN	_	14	case	14:case	_
12	a	a	DET	DT	Definite=Ind|PronType=Art	14	det	14:det	_
13	staff	staff	NOUN	NN	Number=Sing	14	compound	14:compound	_
14	member	member	NOUN	NN	Number=Sing	8	obl	8:obl:by	_
15	but	but	CCONJ	CC	_	20	cc	20:cc	_
16	the	the	DET	DT	Definite=Def|PronType=Art	17	det	17:det	_
17-18	club's	_	_	_	_	_	_	_	_
17	club	club	NOUN	NN	Number=Sing	20	nmod:poss	20:nmod:poss	_
18	's	's	PART	POS	_	17	case	17:case	_
19	actual	actual	ADJ	JJ	Degree=Pos	20	amod	20:amod	_
20	OWNER	owner	NOUN	NN	Number=Sing	3	conj	3:conj:but	SpaceAfter=No
21	?!	?!	PUNCT	.	_	3	punct	3:punct	_

# sent_id = email-enronsent06_01-0061
# text = Attached for your review are copies of the settlement documents that were filed today in the Gas Industry Restructuring/Natural Gas Strategy proceeding, including the Motion for Approval of the Comprehensive Settlement that is supported by thirty signatories to the Comprehensive Settlement, the Comprehensive Settlement document itself, and the various appendices to the settlement.?
1	Attached	attach	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_
2	for	for	ADP	IN	_	4	case	4:case	_
3	your	you	PRON	PRP$	Person=2|Poss=Yes|PronType=Prs	4	nmod:poss	4:nmod:poss	_
4	review	review	NOUN	NN	Number=Sing	1	obl	1:obl:for	_
5	are	be	AUX	VBP	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	1	aux:pass	1:aux:pass	_
6	copies	copy	NOUN	NNS	Number=Plur	1	nsubj:pass	1:nsubj:pass	_
7	of	of	ADP	IN	_	10	case	10:case	_
8	the	the	DET	DT	Definite=Def|PronType=Art	10	det	10:det	_
9	settlement	settlement	NOUN	NN	Number=Sing	10	compound	10:compound	_
10	documents	document	NOUN	NNS	Number=Plur	6	nmod	6:nmod:of|13:nsubj:pass	_
11	that	that	PRON	WDT	PronType=Rel	13	nsubj:pass	10:ref	_
12	were	be	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	13	aux:pass	13:aux:pass	_
13	filed	file	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	10	acl:relcl	10:acl:relcl	_
14	today	today	NOUN	NN	Number=Sing	13	obl:tmod	13:obl:tmod	_
15	in	in	ADP	IN	_	24	case	24:case	_
16	the	the	DET	DT	Definite=Def|PronType=Art	24	det	24:det	_
17	Gas	Gas	PROPN	NNP	Number=Sing	18	compound	18:compound	_
18	Industry	Industry	PROPN	NNP	Number=Sing	19	compound	19:compound	_
19	Restructuring	Restructuring	PROPN	NNP	Number=Sing	24	compound	24:compound	SpaceAfter=No
20	/	/	SYM	,	_	23	cc	23:cc	SpaceAfter=No
21	Natural	Natural	ADJ	NNP	Degree=Pos	22	amod	22:amod	_
22	Gas	Gas	PROPN	NNP	Number=Sing	23	compound	23:compound	_
23	Strategy	Strategy	PROPN	NNP	Number=Sing	19	conj	19:conj|24:compound	_
24	proceeding	proceeding	NOUN	NN	Number=Sing	13	obl	13:obl:in	SpaceAfter=No
25	,	,	PUNCT	,	_	6	punct	6:punct	_
26	including	include	VERB	VBG	VerbForm=Ger	28	case	28:case	_
27	the	the	DET	DT	Definite=Def|PronType=Art	28	det	28:det	_
28	Motion	motion	NOUN	NN	Number=Sing	6	nmod	6:nmod:including|37:nsubj:pass	_
29	for	for	ADP	IN	_	30	case	30:case	_
30	Approval	approval	NOUN	NN	Number=Sing	28	nmod	28:nmod:for	_
31	of	of	ADP	IN	_	34	case	34:case	_
32	the	the	DET	DT	Definite=Def|PronType=Art	34	det	34:det	_
33	Comprehensive	comprehensive	ADJ	JJ	Degree=Pos	34	amod	34:amod	_
34	Settlement	settlement	NOUN	NN	Number=Sing	30	nmod	30:nmod:of	_
35	that	that	PRON	WDT	PronType=Rel	37	nsubj:pass	28:ref	_
36	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	37	aux:pass	37:aux:pass	_
37	supported	support	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	28	acl:relcl	28:acl:relcl	_
38	by	by	ADP	IN	_	40	case	40:case	_
39	thirty	thirty	NUM	CD	NumType=Card	40	nummod	40:nummod	_
40	signatories	signatory	NOUN	NNS	Number=Plur	37	obl	37:obl:by	_
41	to	to	ADP	IN	_	44	case	44:case	_
42	the	the	DET	DT	Definite=Def|PronType=Art	44	det	44:det	_
43	Comprehensive	comprehensive	ADJ	JJ	Degree=Pos	44	amod	44:amod	_
44	Settlement	settlement	NOUN	NN	Number=Sing	40	nmod	40:nmod:to	SpaceAfter=No
45	,	,	PUNCT	,	_	49	punct	49:punct	_
46	the	the	DET	DT	Definite=Def|PronType=Art	49	det	49:det	_
47	Comprehensive	comprehensive	ADJ	JJ	Degree=Pos	49	amod	49:amod	_
48	Settlement	settlement	NOUN	NN	Number=Sing	49	compound	49:compound	_
49	document	document	NOUN	NN	Number=Sing	28	conj	6:nmod:including|28:conj:and|37:nsubj:pass	_
50	itself	itself	PRON	PRP	Gender=Neut|Number=Sing|Person=3|PronType=Prs	49	nmod:npmod	49:nmod:npmod	SpaceAfter=No
51	,	,	PUNCT	,	_	55	punct	55:punct	_
52	and	and	CCONJ	CC	_	55	cc	55:cc	_
53	the	the	DET	DT	Definite=Def|PronType=Art	55	det	55:det	_
54	various	various	ADJ	JJ	Degree=Pos	55	amod	55:amod	_
55	appendices	appendix	NOUN	NNS	Number=Plur	28	conj	6:nmod:including|28:conj:and|37:nsubj:pass	_
56	to	to	ADP	IN	_	58	case	58:case	_
57	the	the	DET	DT	Definite=Def|PronType=Art	58	det	58:det	_
58	settlement	settlement	NOUN	NN	Number=Sing	55	nmod	55:nmod:to	SpaceAfter=No
59	.?	.?	PUNCT	.	_	1	punct	1:punct	_

# sent_id = email-enronsent27_01-0014
# text = They are kind of in rank order but as I stated if I find the piece that I like we will purchase it.
1	They	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	7	nsubj	7:nsubj	_
2	are	be	AUX	VBP	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	7	cop	7:cop	_
3	kind	kind	NOUN	NN	ExtPos=ADV|Number=Sing	7	obl:npmod	7:obl:npmod	_
4	of	of	ADP	IN	_	3	fixed	3:fixed	_
5	in	in	ADP	IN	_	7	case	7:case	_
6	rank	rank	NOUN	NN	Number=Sing	7	compound	7:compound	_
7	order	order	NOUN	NN	Number=Sing	0	root	0:root	_
8	but	but	CCONJ	CC	_	22	cc	22:cc	_
9	as	as	SCONJ	IN	_	11	mark	11:mark	_
10	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	11	nsubj	11:nsubj	_
11	stated	state	VERB	VBD	Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin	22	advcl	22:advcl:as	_
12	if	if	SCONJ	IN	_	14	mark	14:mark	_
13	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	14	nsubj	14:nsubj	_
14	find	find	VERB	VBP	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin	22	advcl	22:advcl:if	_
15	the	the	DET	DT	Definite=Def|PronType=Art	16	det	16:det	_
16	piece	piece	NOUN	NN	Number=Sing	14	obj	14:obj|19:obj	_
17	that	that	PRON	WDT	PronType=Rel	19	obj	16:ref	_
18	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	19	nsubj	19:nsubj	_
19	like	like	VERB	VBP	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin	16	acl:relcl	16:acl:relcl	_
20	we	we	PRON	PRP	Case=Nom|Number=Plur|Person=1|PronType=Prs	22	nsubj	22:nsubj	_
21	will	will	AUX	MD	VerbForm=Fin	22	aux	22:aux	_
22	purchase	purchase	VERB	VB	VerbForm=Inf	7	conj	7:conj:but	_
23	it	it	PRON	PRP	Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs	22	obj	22:obj	SpaceAfter=No
24	.	.	PUNCT	.	_	7	punct	7:punct	_

# sent_id = 1
# text = Masha bought a frying pan, and the boys bought vegetables
1	Masha	Masha	PROPN	NNP	Number=Sing	2	nsubj	_	_
2	bought	buy	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	_	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	5	det	_	_
4	frying	frying	VERB	VBG	VerbForm=Ger	5	compound	_	_
5	pan	pan	NOUN	NN	Number=Sing	2	obj	_	SpaceAfter=No
6	,	,	PUNCT	,	_	10	punct	_	_
7	and	and	CCONJ	CC	_	10	cc	_	_
8	the	the	DET	DT	Definite=Def|PronType=Art	9	det	_	_
9	boys	boy	NOUN	NNS	Number=Plur	10	nsubj	_	_
10	bought	buy	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	2	conj	_	_
11	vegetables	vegetable	NOUN	NNS	Number=Plur	10	obj	_	SpaceAfter=No

# sent_id = email-enronsent16_01-0041
# text = This would have to be determined on a case by case basis.
1	This	this	PRON	DT	Number=Sing|PronType=Dem	3	nsubj	3:nsubj|6:nsubj:xsubj	_
2	would	would	AUX	MD	VerbForm=Fin	3	aux	3:aux	_
3	have	have	VERB	VB	VerbForm=Inf	0	root	0:root	_
4	to	to	PART	TO	_	6	mark	6:mark	_
5	be	be	AUX	VB	VerbForm=Inf	6	aux:pass	6:aux:pass	_
6	determined	determine	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	3	xcomp	3:xcomp	_
7	on	on	ADP	IN	_	12	case	12:case	_
8	a	a	DET	DT	Definite=Ind|PronType=Art	12	det	12:det	_
9	case	case	NOUN	NN	Number=Sing	12	compound	12:compound	_
10	by	by	ADP	IN	_	11	case	11:case	_
11	case	case	NOUN	NN	Number=Sing	9	nmod	9:nmod:by	_
12	basis	basis	NOUN	NN	Number=Sing	6	obl	6:obl:on	SpaceAfter=No
13	.	.	PUNCT	.	_	3	punct	3:punct	_

# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0021
# text = And to those who don't even know their crimes, not even that.
1	And	and	CCONJ	CC	_	14	cc	14:cc	_
2	to	to	ADP	IN	_	3	case	3:case	_
3	those	that	PRON	DT	Number=Plur|PronType=Dem	14	nmod	8:nsubj|14:nmod:to	_
4	who	who	PRON	WP	PronType=Rel	8	nsubj	3:ref	_
5-6	don't	_	_	_	_	_	_	_	_
5	do	do	AUX	VBP	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	8	aux	8:aux	_
6	n't	not	PART	RB	_	8	advmod	8:advmod	_
7	even	even	ADV	RB	_	8	advmod	8:advmod	_
8	know	know	VERB	VB	VerbForm=Inf	3	acl:relcl	3:acl:relcl	_
9	their	they	PRON	PRP$	Number=Plur|Person=3|Poss=Yes|PronType=Prs	10	nmod:poss	10:nmod:poss	_
10	crimes	crime	NOUN	NNS	Number=Plur	8	obj	8:obj	SpaceAfter=No
11	,	,	PUNCT	,	_	14	punct	14:punct	_
12	not	not	PART	RB	_	14	advmod	14:advmod	_
13	even	even	ADV	RB	_	14	advmod	14:advmod	_
14	that	that	DET	DT	Number=Sing|PronType=Dem	0	root	0:root	SpaceAfter=No
15	.	.	PUNCT	.	_	14	punct	14:punct	_

# sent_id = 2
# text = Masha bought vegetables, and the boys bought a frying pan
1	Masha	Masha	PROPN	NNP	Number=Sing	2	nsubj	_	_
2	bought	buy	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	_	_
3	vegetables	vegetable	NOUN	NNS	Number=Plur	2	obj	_	SpaceAfter=No
4	,	,	PUNCT	,	_	8	punct	_	_
5	and	and	CCONJ	CC	_	8	cc	_	_
6	the	the	DET	DT	Definite=Def|PronType=Art	7	det	_	_
7	boys	boy	NOUN	NNS	Number=Plur	8	nsubj	_	_
8	bought	buy	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	2	conj	_	_
9	a	a	DET	DT	Definite=Ind|PronType=Art	11	det	_	_
10	frying	fry	VERB	VBG	VerbForm=Ger	11	amod	_	_
11	pan	pan	NOUN	NN	Number=Sing	8	obj	_	SpaceAfter=No

# newdoc id = reviews-190256
# sent_id = reviews-190256-0001
# newpar id = reviews-190256-p0001
# text = I was thoroughly impressed!
1	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	4	nsubj	4:nsubj	_
2	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	4	cop	4:cop	_
3	thoroughly	thoroughly	ADV	RB	_	4	advmod	4:advmod	_
4	impressed	impressed	ADJ	JJ	Degree=Pos	0	root	0:root	SpaceAfter=No
5	!	!	PUNCT	.	_	4	punct	4:punct	_

'''
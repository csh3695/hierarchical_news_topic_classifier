# Hierarchical News Topic Classifier

- Topic Tree 기반 News Topic Classifier입니다.

- Sample Results(`Threshold=0.5`)



        [단독]농협은행, 올해 11월까지 주택담보대출 전면 중단 
            Greedy Prediction: ['경제', '금융_재테크']
            Threshold Prediction: [['경제', '금융_재테크']]
            Top10 Most Focused Tokens: [('규제', 0.02074139006435871), ('높', 0.019017456099390984), ('##상', 0.015527249313890934), ('나라', 0.015339044854044914), ('은행', 0.015131342224776745), ('은행', 0.013206366449594498), ('은행', 0.0127566484734416), ('후보자', 0.012473318725824356), ('##은행', 0.01234383787959814), ('##은행', 0.012075147591531277)]
            Top10 Less Focused Tokens: [('이', 0.0006344065186567605), ('##이', 0.0006556065054610372), ('##지', 0.0006778125534765422), ('19', 0.0006792834610678256), ('가계', 0.0007239215774461627), ('##의', 0.0007485878886654973), ('##월', 0.0007827666122466326), ('목표치', 0.0008124701562337577), ('##대', 0.0008175603579729795), ('가계', 0.0008429849985986948)]
        
        코스피, 상반기 매출 첫 1,000조 벽 넘었다
            Greedy Prediction: ['경제', '산업_기업']
            Threshold Prediction: [['경제', '증권_증시'], ['경제', '산업_기업']]
            Top10 Most Focused Tokens: [('기준', 0.029322784394025803), ('기준', 0.02593935839831829), ('바뀌', 0.020273717120289803), ('기준', 0.020096587017178535), ('수출', 0.014988172799348831), ('발표', 0.013936440460383892), ('분류', 0.013568257912993431), ('매출', 0.012913146987557411), ('상반기', 0.012409815564751625), ('분석', 0.012319570407271385)]
            Top10 Less Focused Tokens: [(',', 7.35385183361359e-05), (',', 9.422071889275685e-05), ('##의', 0.00014748857938684523), ('원', 0.00016008343663997948), ('##들의', 0.00017771180137060583), ('000', 0.0002206223871326074), ('원', 0.00026493618497624993), ('1', 0.0002659692254383117), ('##조', 0.00028879629098810256), ('000', 0.00036633299896493554)]
        
        서울시, 10월부터 배달 노동자 상해보험료 전액 지원
            Greedy Prediction: ['사회', '의료_건강']
            Threshold Prediction: [['사회', '의료_건강'], ['사회', '노동_복지']]
            Top10 Most Focused Tokens: [('노동자', 0.02544989623129368), ('노동자', 0.024628663435578346), ('노동자', 0.02414230816066265), ('노동자', 0.023878011852502823), ('노동자', 0.022483637556433678), ('노동자', 0.02192850224673748), ('##업체', 0.02166169136762619), ('노동자', 0.021094761788845062), ('노동자', 0.020690500736236572), ('노동자', 0.02063259854912758)]
            Top10 Less Focused Tokens: [('지난', 0.0004498145426623523), ('상해', 0.0004513434541877359), ('.', 0.000476459797937423), ('##가', 0.0005186215857975185), ("'", 0.0005222844774834812), ('상해', 0.0005383904208429158), ('상해', 0.0005630490486510098), ('상해', 0.0006635786266997457), ('이', 0.0006893587415106595), ('##월', 0.0006905414629727602)]
        
        [날씨] 선선해진 아침저녁 공기…전국 곳곳 소나기
            Greedy Prediction: ['사회', '사회일반']
            Threshold Prediction: [['사회', '사회일반']]
            Top10 Most Focused Tokens: [('선선', 0.04104120656847954), ('##안', 0.03221028670668602), ('공기', 0.03196129947900772), ('19', 0.022761110216379166), ('##저', 0.022340739145874977), ('##요', 0.021669015288352966), ('기상', 0.021210599690675735), ('##녁', 0.02105720154941082), ('##니다', 0.019796131178736687), ('##니다', 0.01940329000353813)]
            Top10 Less Focused Tokens: [('##에', 0.0006233122549019754), ('##을', 0.0006901170127093792), ('##와', 0.0007742101443000138), ('##와', 0.0008444902487099171), ('##처럼', 0.0008932395139709115), ('##으로', 0.000992782530374825), ('##씩', 0.0010131351882591844), ('##부터', 0.0010908294934779406), ('##에', 0.0011847252026200294), ('##에', 0.0011976772220805287)]
        
        크래프톤, '배틀그라운드: 뉴 스테이트' 앱스토어 사전등록 시작 
            Greedy Prediction: ['IT_과학', '인터넷_SNS']
            Threshold Prediction: [['IT_과학', '인터넷_SNS'], ['IT_과학', '콘텐츠'], ['IT_과학', '모바일']]
            Top10 Most Focused Tokens: [('##한다', 0.0458141528069973), ('##한다', 0.04454297199845314), ('##됐', 0.041181039065122604), ('##했', 0.03729099780321121), ('##며', 0.03345058858394623), ('경험', 0.032172996550798416), ('기준', 0.030952848494052887), ('##할', 0.029505405575037003), ('밝혔', 0.029471447691321373), ('##다', 0.027414420619606972)]
            Top10 Less Focused Tokens: [('##그라', 0.0005521262064576149), ('##그라', 0.0005650147213600576), ('##그라', 0.0005960051785223186), ('##그라', 0.0006182691431604326), ('크래', 0.0006550229736603796), ('##그라', 0.0007193479686975479), ('##프', 0.0007217125967144966), ('크래', 0.0007578434888273478), ("'", 0.0007709715282544494), ('##프', 0.0007999952067621052)]
        
        우체국노조 "우본 사회적 합의 부정"···무기한 농성
            Greedy Prediction: ['사회', '노동_복지']
            Threshold Prediction: [['사회', '노동_복지']]
            Top10 Most Focused Tokens: [('농성', 0.036791179329156876), ('농성', 0.03273558244109154), ('투쟁', 0.03268267586827278), ('노동자', 0.029353275895118713), ('노조', 0.026314472779631615), ('계약', 0.026179898530244827), ('노동자', 0.025753973051905632), ('##노조', 0.02490578591823578), ('노조', 0.022197909653186798), ('##노동', 0.021241825073957443)]
            Top10 Less Focused Tokens: [('·', 0.0007306637708097696), ('##우', 0.0008194688707590103), ('·', 0.0008265115902759135), ('##은', 0.0008438610238954425), ('·', 0.0008560080896131694), ('##우', 0.0008710863185115159), ('는', 0.0010493247536942363), ('##편', 0.0010800424497574568), ('"', 0.0010870343539863825), ("'", 0.0010945644462481141)]

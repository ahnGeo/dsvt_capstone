# dsvt_capstone

##### Problem definition
   Self-supervised learning은 데이터셋의 정답 레이블을 사용하지 않고 데이터 자체로 representation learning을 하는 방법을 말한다. 근래 등장한 self-supervised 비디오 모델들은 supervised learning을 사용한 모델들을 웃도는 classification 정확도와 좋은 일반화 능력을 가지고 있어 주목받고 있다. 이를 이용하여 well-pretrained self-supervised video model을 구축하고 이를 미세하게 조정하여 여러 종류의 downstream task에 성공적으로 적용하기 위한 연구들이 진행되고 있다.   
   Action recognition은 주어진 비디오가 어떤 action 에 해당하는지 맞추는 classification task이다. Action recognition은 주로 3~10초의 비디오안에서 시간에 따라 변하는 물체의 움직임을 포착하는 것이 중요한 task이다. Action recognition을 위한 비디오 모델에게는 각 프레임의 spatial information과 전반적인 비디오의 temporal action information에 대한 feature를 잘 추출하고 적절하게 조합하는 능력이 요구된다. 현재까지 여러 좋은 비디오 모델들이 Kinetics400, Something-something, UCF101 등 주요한 비디오 데이터셋에서 높은 classification accuracy를 보여주고 있다. 그러나 pretrain에 사용한 dataset과 다른 도메인을 가진 downstream dataset에서는 성능이 급격하게 하락한다.  
 본 연구는 이를 기존의 self-supervised video representation learning model들이 pretrain하는 과정에서 action 자체가 아닌 background scene bias에만 집중하기 때문이라고 주장한다. 그리고 이를 개선하기 위해 foreground와 background를 구분해가며 pretrain을 진행해 downstream에서 action을 이해하는데 중요한 foreground에 대한 정보만을 효과적으로 사용하는 DSVT 모델을 제안한다. 

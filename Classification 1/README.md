# Predicting Customer Satisfaciton Santander Dataset 

This project aims to predict the customer satisfaciton 0 or 1. I applied bunch of techniques to prepare the data. 


---

## Overview

dataset was imbalanced and high dimensional (400000, 202) and it was not possible to understand by just looking at it it is so complex: like this : (ID_code	target	var_0	var_1	var_2	var_3	var_4	var_5	var_6	var_7	var_8	var_9	var_10	var_11	var_12	var_13	var_14	var_15	var_16	var_17	var_18	var_19	var_20	var_21	var_22	var_23	var_24	var_25	var_26	var_27	var_28	var_29	var_30	var_31	var_32	var_33	var_34	var_35	var_36	var_37	var_38	var_39	var_40	var_41	var_42	var_43	var_44	var_45	var_46	var_47	var_48	var_49	var_50	var_51	var_52	var_53	var_54	var_55	var_56	var_57	var_58	var_59	var_60	var_61	var_62	var_63	var_64	var_65	var_66	var_67	var_68	var_69	var_70	var_71	var_72	var_73	var_74	var_75	var_76	var_77	var_78	var_79	var_80	var_81	var_82	var_83	var_84	var_85	var_86	var_87	var_88	var_89	var_90	var_91	var_92	var_93	var_94	var_95	var_96	var_97	var_98	var_99	var_100	var_101	var_102	var_103	var_104	var_105	var_106	var_107	var_108	var_109	var_110	var_111	var_112	var_113	var_114	var_115	var_116	var_117	var_118	var_119	var_120	var_121	var_122	var_123	var_124	var_125	var_126	var_127	var_128	var_129	var_130	var_131	var_132	var_133	var_134	var_135	var_136	var_137	var_138	var_139	var_140	var_141	var_142	var_143	var_144	var_145	var_146	var_147	var_148	var_149	var_150	var_151	var_152	var_153	var_154	var_155	var_156	var_157	var_158	var_159	var_160	var_161	var_162	var_163	var_164	var_165	var_166	var_167	var_168	var_169	var_170	var_171	var_172	var_173	var_174	var_175	var_176	var_177	var_178	var_179	var_180	var_181	var_182	var_183	var_184	var_185	var_186	var_187	var_188	var_189	var_190	var_191	var_192	var_193	var_194	var_195	var_196	var_197	var_198	var_199
0	train_0	0.0	8.9255	-6.7863	11.9081	5.0930	11.4607	-9.2834	5.1187	18.6266	-4.9200	5.7470	2.9252	3.1821	14.0137	0.5745	8.7989	14.5691	5.7487	-7.2393	4.2840	30.7133	10.5350	16.2191	2.5791	2.4716	14.3831	13.4325	-5.1488	-0.4073	4.9306	5.9965	-0.3085	12.9041	-3.8766	16.8911	11.1920	10.5785	0.6764	7.8871	4.6667	3.8743	-5.2387	7.3746	11.5767	12.0446	11.6418	-7.0170	5.9226	-14.2136	16.0283	5.3253	12.9194	29.0460	-0.6940	5.1736	-0.7474	14.8322	11.2668	5.3822	2.0183	10.1166	16.1828	4.9590	2.0771	-0.2154	8.6748	9.5319	5.8056	22.4321	5.0109	-4.7010	21.6374	0.5663	5.1999	8.8600	43.1127	18.3816	-2.3440	23.4104	6.5199	12.1983	13.6468	13.8372	1.3675	2.9423	-4.5213	21.4669	9.3225	16.4597	7.9984	-1.7069	-21.4494	6.7806	11.0924	9.9913	14.8421	0.1812	8.9642	16.2572	2.1743	-3.4132	9.4763	13.3102	26.5376	1.4403	14.7100	6.0454	9.5426	17.1554	14.1104	24.3627	2.0323	6.7602	3.9141	-0.4851	2.5240	1.5093	2.5516	15.5752	-13.4221	7.2739	16.0094	9.7268	0.8897	0.7754	4.2218	12.0039	13.8571	-0.7338	-1.9245	15.4462	12.8287	0.3587	9.6508	6.5674	5.1726	3.1345	29.4547	31.4045	2.8279	15.6599	8.3307	-5.6011	19.0614	11.2663	8.6989	8.3694	11.5659	-16.4727	4.0288	17.9244	18.5177	10.7800	9.0056	16.6964	10.4838	1.6573	12.1749	-13.1324	17.6054	11.5423	15.4576	5.3133	3.6159	5.0384	6.6760	12.6644	2.7004	-0.6975	9.5981	5.4879	-4.7645	-8.4254	20.8773	3.1531	18.5618	7.7423	-10.1245	13.7241	-3.5189	1.7202	-8.4051	9.0164	3.0657	14.3691	25.8398	5.8764	11.8411	-19.7159	17.5743	0.5857	4.4354	3.9642	3.1364	1.6910	18.5227	-2.3978	7.8784	8.5635	12.7803	-1.0914
)

---

## Key Steps

### 1. assess_linearity function  to decide if this highly non interpretable data is non linear or linear. 


![PCA Visualization](screenshots/PCA.png)

---

### 2. There was imbalance so i applied smote to balance the dataset counter({0.0: 179902, 1.0: 20098})

---

### 3. QuantileTransformer scaler for distirbution (normal) scaler = QuantileTransformer(output_distribution='normal', random_state=42)
did not use standart scaler because of the reasons the quantile is better 


### 4. deep learning model 
here is the architecture: 

here is the model performance in the initial steps of the epochs (model converges fast):


here is the last epochs The accuracy is almost %92:



---

### 5. PRediciton results of the model
After accomplishing all of the steps this is the result we got from the model we built. 

![Model Performance](screenshots/model_performance.png)

---

## Conclusion

I applied bunch of techniques to nmanipulate data and infer meaninng from it 

**Potential Next Steps**  


from lime.lime_text import LimeTextExplainer

def plot_lime(test_clean_text,nb_pipeline,vec):
  test_vector = vec.transform([test_clean_text])

  class_names = [0,1]
  explainer2 = LimeTextExplainer(class_names=class_names)

  exp = explainer2.explain_instance(test_clean_text, nb_pipeline.predict_proba, num_features=10, labels=[0, 1])
  # print('Predicted class =', class_names[nb_pipeline[1].predict(test_vector).reshape(1,-1)[0,0]])
  # print(nb_pipeline.predict_proba([test_clean_text]).round(3))
  return exp

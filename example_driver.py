import Word2Vec

# train example
fn = "harryPotter1.txt"
txt = Word2Vec.normalize_text(fn)
model = Word2Vec.SkipGram(txt)  # first time using this function may take a moment
model.learn_embeddings(epochs=3)

# example of similarity check
print("harry", "potter", model.compute_similarity("harry", "potter"))
print("magic", "wand", model.compute_similarity("magic", "wand"))
print("hat", "wizard", model.compute_similarity("hat", "wizard"))
print("harry", "gryffindor", model.compute_similarity("harry", "gryffindor"))
print("gryffindor", "hufflepuff", model.compute_similarity("gryffindor", "hufflepuff"))

# example of load model
model = Word2Vec.load_model("harrypotter.pickle")

# example of starting semantle game
semantle = Word2Vec.SemantleSolver(model)
semantle.set_target_word(
    "harry"
)  # you can choose your target word for testing, not necessarily
semantle.semantle_game()

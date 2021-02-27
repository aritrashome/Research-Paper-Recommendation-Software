import nltk.chunk
import itertools
import nltk.corpus, nltk.tag

def conll_tag_chunks(chunk_sents):
    tag_sents = [nltk.chunk.tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in chunk_tags] for chunk_tags in tag_sents]

def ubt_conll_chunk_accuracy(train_sents, test_sents):
    train_chunks = conll_tag_chunks(train_sents)
    test_chunks = conll_tag_chunks(test_sents)

    u_chunker = nltk.tag.UnigramTagger(train_chunks)

    ub_chunker = nltk.tag.BigramTagger(train_chunks, backoff=u_chunker)

    ubt_chunker = nltk.tag.TrigramTagger(train_chunks, backoff=ub_chunker)

    ut_chunker = nltk.tag.TrigramTagger(train_chunks, backoff=u_chunker)

    utb_chunker = nltk.tag.BigramTagger(train_chunks, backoff=ut_chunker)


conll_train = nltk.corpus.conll2000.chunked_sents('train.txt')
conll_test = nltk.corpus.conll2000.chunked_sents('test.txt')
ubt_conll_chunk_accuracy(conll_train, conll_test)

# treebank chunking accuracy test
treebank_sents = nltk.corpus.treebank_chunk.chunked_sents()
ubt_conll_chunk_accuracy(treebank_sents[:2000], treebank_sents[2000:])


class TagChunker(nltk.chunk.ChunkParserI):
    def __init__(self, chunk_tagger):
        self._chunk_tagger = chunk_tagger

    def parse(self, tokens):
        # split words and part of speech tags
        (words, tags) = zip(*tokens)
        # get IOB chunk tagsTagChunker
        chunks = self._chunk_tagger.tag(tags)
        # join words with chunk tags
        wtc = itertools.izip(words, chunks)
        # w = word, t = part-of-speech tag, c = chunk tag
        lines = [' '.join([w, t, c]) for (w, (t, c)) in wtc if c]
        # create tree from conll formatted chunk lines
        return nltk.chunk.conllstr2tree('\n'.join(lines))

# sentence should be a list of words
sentence=u'We saw the yellow dog'
tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
tree = chunker.parse(tagged)
# for each noun phrase sub tree in the parse tree
for subtree in tree.subtrees(filter=lambda t: t.node == 'NP'):
    # print the noun phrase as a list of part-of-speech tagged words
    print subtree.leaves()
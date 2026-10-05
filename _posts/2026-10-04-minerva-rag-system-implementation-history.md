---
layout: post
title: "Minerva RAG System: Implementation History"
---

Minerva is a RAG system for my personal notes. Indexing and search run entirely on my Mac with small local models. AI agents query it through an MCP server. The agent can be a cloud application like Claude Desktop, or a model running locally in LM Studio. This post tells how I built it twice: first by letting an AI agent decide almost everything, then again from scratch, measuring every choice. You will find failures (many), a few victories and what I learned about building RAG with small local models.

## First Legacy Minerva

It was autumn 2025 and, after reading and studying, I wanted to explore the world of AI-based applications.
There were many AI-related tools that I could build but my crux has always been finding information: I stuff information in my notes and then I always struggle to find it, so a [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) system seemed the perfect tool for me.

I had no idea how to build RAG systems, I had only read some articles here and there.
In order to see one at work I just told my AI agent to build one. I used Claude Code with a [cool tool built by a colleague, named dev-workflow](https://github.com/andreaserradev-gbj/dev-workflow). It took me one month.

The AI agent decided most of the details. I picked the technology (Python), the database (Chroma: no servers, no remote databases, just plain files) and the overall flow, which had two steps. First, extractors converted data from each native source into a common intermediate format (markdown + json). Then a common ingestor chunked the notes, embedded the chunks and stored them in Chroma.

Search was a dense vector search on Chroma's [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) index. Embeddings came from Ollama at first, and later also from LM Studio, OpenAI and Gemini.

I built an [MCP server](https://modelcontextprotocol.io/); at the beginning, I tried to use it with a custom implemented chat but I wasn't satisfied so I went back to MCP only, using it with AI desktop applications like Claude Desktop; it always found what I was looking for and I was satisfied (when I analytically evaluated it, I discovered it wasn't _that_ good, read below!).

I published Minerva in December 2025 [on GitHub](https://github.com/michele-orsini-71/minerva).

## Minerva2 project starts

After a few months (it was March 2026) I already knew more about RAG systems. I had read [Contextual Retrieval in AI Systems](https://www.anthropic.com/engineering/contextual-retrieval) and Chapter 6, "RAG and Agents", of _AI Engineering_ by Chip Huyen (O'Reilly, 2025). Read it, it is enlightening. How little I knew about RAG systems when I built Minerva!! It was time for a Minerva on steroids.

The article describes a technique I will call contextualization. Before indexing, an LLM reads the whole document and writes a short context for each chunk. This context is added to the chunk before it is embedded and indexed. It helps when a chunk does not name its subject, or refers to something defined elsewhere in the document, like "this cat" or "the previous quarter".

The Anthropic article claimed: 35% fewer retrieval failures with contextual embeddings, and 67% fewer with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and a reranker added, it looked promising. A reranker is a model that re-scores the top results by reading the query and each chunk together.

I had some clear ideas in mind

- local: indexing and search run on my Mac, using only local models. I had three reasons: privacy, working without an internet connection, and no costs per call. Legacy Minerva was different: it could also use remote embedding services like OpenAI and Gemini.
- no more two steps, bad UX, I wanted a single operation, without intermediate storage, from the original data to the embeddings in the database
- json configuration files of legacy Minerva were a mess and the whole system was too complex to configure to the point that I needed to add two helpers `minerva-doc` and `minerva-kb` to simplify the user's life
- no more Python, I wanted non-dev people to be able to download a package and install it without having to deal with git, pip/pipx and the terminal if possible
- what about having a UI app to manage everything? Cool!

For this reason, I decided to use C#/.NET. I also thought that coding would be simpler:

- readability
- strong typing
- has lots of libraries and frameworks, including the official MCP/AI SDKs, to speed up development

Then, still with Chroma? No, I wanted to use a full-fledged DB that could provide me both vector and keyword search, so I went with PostgreSQL + pgvector + built-in full-text search (please note, not BM25 as in the Anthropic article): both "legs" of retrieval, lexical (full-text) and dense vectors in one shot. Was it a wise choice? More on this later.
Actually, a C# feasibility study recommended Qdrant, but I rejected it because it needs Docker or the cloud. PostgreSQL wasn't an easy install either, but it seemed manageable. I also wanted to learn how to use it. Looking back, this is where I lost sight of one of my goals, a simple installation: the chance to learn something new fascinated me more.

At this point, I was ready to implement Anthropic's recipe. The plan was:

- during ingestion, add a context to each chunk (contextualization); this helps both the full-text and the semantic search
- at query time, run a keyword (full-text) search
- run a semantic search with embeddings
- merge the two ranked lists with RRF
- later, add a reranker

RRF ([Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)) ignores the scores of the two searches and uses only the position of each chunk in each list. In Minerva I used a weighted version:

`score(chunk) = α/(k + rank_vector) + (1 − α)/(k + rank_fts)`

- `rank_vector`, `rank_fts`: the chunk's position in the vector list and in the full-text list
- `k`: the RRF constant, fixed at 60
- `α` (alpha, HybridAlpha): the weight of the vector list. `α` = 1 means pure vector search, `α` = 0 means pure full-text search, `α` = 0.5 gives both equal weight.

## A first clash with reality

Around April-May 2026 I had my first implementation:

- a Core library
- an indexer CLI tool
- a CLI search command to verify query results

Cool, but some of my previous decisions came back to bite me.

The easier problem came first. I had to drop Microsoft's Host and dependency injection (DI) framework, because it prevented me from building a solid preflight check. The idea was to verify all prerequisites before indexing starts, and fail with a meaningful error if one is missing. As expected, the framework decided when to create objects, and object creation failed when a setting was wrong. So I had to check the settings myself first, and then create the objects explicitly.

The real pain came from my poor knowledge of the limits of local models. Want to build a local system? Do you have the slightest idea how poor the performance of local models is? Are you aware that your MacBook will suffer with all those inference calls? Let's start.

First, contextualization was not done as Anthropic described because I cannot send a whole document to a small model and expect it to be able to ingest it. So I had to split big docs into fragments. Then build a summary of the fragments and finally ask for a contextualization of the chunk using this summary. Not exactly what was recommended by the article, I didn't know it yet, but this compromise would come back.

Then came ingestion time. With contextualization, 99% of it was spent on LLM calls. Here are the ingestion times for my personal notes (~1.8k docs), which I used as a first sample for indexing:

- full-text search only: ~10 min
- contextualization using a very small model (Qwen2.5-0.5B) 10.7 h
- contextualization using a small model (Gemma 4 E4B) 52.4 h

The problem was clear: each doc was split into chunks, chunks were around ten thousand, and every chunk had its LLM call for contextualization and this required 1-20s, depending on the model used.

This put the idea of a local system at risk! How could I proceed? Then, well, _measuring_: didn't that lovely book about AI Engineering always talk about building evaluation pipelines? Let's do it.

## OMG, building an eval pipeline is so hard!

The long path towards having my eval pipeline started in May and lasted 4 months (holidays and interruptions due to frustrations included).

What are the ingredients for an eval pipeline?

1. corpus: the set of documents to index and to query against
2. annotated queries: each query to the RAG system comes with its "gold source", the right document to return (why not the right chunk? see below)
3. eval runner, which I call the bench: a program that runs all the queries automatically, optionally with different search parameters, and stores the results on disk
4. run analyzer: a tool that reads the results and extracts insights

I started by defining the format of the queries. Then I wrote the bench. It runs "sweeps": it repeats all the queries with different search parameters, which it reads from a TOML file. The parameters were top-k (how many results to return), alpha (the weight of the vector list in the hybrid search) and, later, a switch to turn the reranker on or off. Very cool.

At that point, what I needed was a dataset and the queries and these, of course, must be built in conjunction.

First I had to pick a corpus. Previously I was using my own notes but I wanted something that someone, downloading my repository, should be able to experience themselves, so I used Wikipedia articles, it is possible to download a versioned ZIM file containing a snapshot of a Wikipedia article collection from Kiwix servers.

With all the moving parts around, I decided to use articles, not chunks, as gold sources; otherwise I would have been forced to redefine gold sources as soon as the corpus or chunking changed. It was a shortcut, actually: the RAG returns the best matching chunk and I only verify that this belonged to the source document I used as gold.

Always with compute time in mind, I started extracting docs from the ZIM file, adding them to the text collection and writing queries.

First run of the bench: all gold sources found. That's too easy: how could I make it more difficult? If I wanted to see improvements in future development, I needed to build more difficult to find queries.

I recrafted them so they did not reference words in title or in the document corpus but expressed the same concepts with different words. It was not easy because several times I went too far, for example linking concepts too loosely: only a "hopping" RAG system would have caught them (but this was not in the scope of Minerva, see below) and I had to rewrite or remove them.

Later, but it is worth mentioning it here where we are talking about corpus and queries, I also added queries that mention acronyms or codes, things that a semantic engine may not store and that could be caught by the other leg of hybrid search (that's the reason why hybrid search has a full-text leg).

To read the bench results, I built a Jupyter notebook that I still use today. It is called "runs_comparer" and (surprise!) it compares eval runs. It shows their differences in:

- [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) at K (R@K)
- [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) at K (MRR@K)
- covering rank

R@K is the share of gold sources found in the first K results. For example, R@10 looks at the first 10 results.

MRR@K rewards finding the gold source early. For each query, take 1/rank of the first gold source within the first K results, or 0 if it is missing. Then average over all queries. MRR@10 = 1 means the gold source was always first.

Covering rank is the number of results a client has to read to find all the gold sources of a query.

It also shows the rank of each gold source for every query and every sweep setting, and it lists the "late" gold sources, the ones ranked lowest. Awesome.

N.B. The numbers in this post come from different corpus sizes and settings, so compare them only within the same paragraph. Ingestion times are wall-clock times on the Mac I also use for work, so they are rough estimates.

The other way to make the bench harder was to add _more_ documents. I started by adding random Wikipedia articles as "distractors", but it didn't work (in the meantime, the corpus had become bigger). Random articles had little in common with the gold sources, so they did not pull the search away from the right answer.

So I built a noise builder: a tool that picks the Wikipedia articles semantically closest to each gold source and adds them to the corpus. It worked! Adding 156 neighbour articles dropped MRR@10 from 0.828 to 0.758 while recall held. The gold sources were still found, but ranked lower.

In the end, the corpus and the eval reached their current size:

- corpus: 1283 docs, split into 72,252 chunks. I call it wp1283.
- queries: 84

I also built a small subset for smoke tests (quick checks that nothing is broken): 111 docs and the 18 most difficult queries. I called it wp111.

Do you remember how long it took to index my notes? About 11 hours with one LLM, about 50 with the other. With these big Wikipedia articles, things got much worse. Gemma needed roughly one night to contextualize 10 docs, and Qwen was "only" ten times faster. Yes, you read it right: ingesting the test corpus with the fastest model took almost a week, with my poor Mac crunching night and day, fans at full speed like a helicopter in my office. With Gemma, I estimated around 1000 hours. On my notes I hadn't seen big differences between the two models, so I went with Qwen only and put Gemma aside for the moment. Don't blame me for having "too few" queries and docs, please!


## Contextualization disappointment but wait ...

Finally, I was able to run the first evals and ... it turned out that contextualization was adding _nothing_ to the RAG performance. How was it possible? Was all that compute time wasted?
The contextualization analysis scared me: experimenting with it would mean reindexing the corpus several times. I needed a new idea, so I set contextualization aside for a while and moved on to other features in my roadmap.

The most promising next step was adding a reranker. I expected a reranker to improve Minerva's performance and give me a boost that could make contextualization useless. I added the reranker but, to my disappointment, it only gave some gain, less than promised, with a few strange regressions.

"Fortunately" another problem distracted me from this *double* disappointment! Upon careful inspection I found several flaws in the keyword search as well.

First problem: keyword search expects keywords, but people write natural language queries. If every term is required (AND), a query like "how do cats hunt at night" matches almost no chunk. So I combined all the terms of the query with OR: a chunk matches if it contains any of them.

Second problem: the OR interacted badly with my full-text settings. They were language neutral, because I wanted to index docs in any language. This meant no [stop words](https://en.wikipedia.org/wiki/Stop_word) removal: common words like "how", "do" and "at" stayed in the query, and with OR they matched almost every chunk. It also meant no [stemming](https://en.wikipedia.org/wiki/Stemming): "hunt" did not match "hunting", so the relevant chunks lost part of their score. On top of that, PostgreSQL's ranking function, `ts_rank`, does not consider how rare a word is in the corpus. A chunk full of common words could rank above the chunk containing the rare, meaningful one. The result was a lot of noise.

In this period I was reading some PostgreSQL docs and I wanted to run an experiment writing some Python throw-away code; soon the experiment became an exploration of full-text search capabilities.

I wanted to overcome the two problems found in the full-text search leg and I tried different strategies.

I tried to use the built-in full-text search (the same I was using in Minerva) but with trigram matching to 'simulate' stemming, but I still had that noise problem from the 'OR' and the stop words: I really needed [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

So, I tried a real BM25 engine (ParadeDB `pg_search` was my choice). Using BM25 means using a [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) and thanks to IDF it was able to ignore all the frequent-but-irrelevant terms. This solved the 'OR' + stop words problem but not the stemming one.

To overcome it, I also tried fuzzy matching search with BM25 (trigram indexes didn't fit well in Minerva ingestion pipeline), but short common words matched many variants, so the results were filled with noise and I decided to live without it.

Fortunately, a consideration gave me some relief: wasn't BM25 used mostly to detect keywords and acronyms that are not caught by the semantic engine? Then the stemming problem was not that important.

Given these premises, I felt ready for a new round of changes in Minerva.

I switched from `ts_rank` to a real BM25 with the same ParadeDB `pg_search` I discovered in my previous exploration, and the full-text leg of my hybrid search improved considerably: at alpha = 0.3, that is when the weight of full-text search is higher, R@5 went from 0.71 to 0.87 and MRR@10 from 0.60 to 0.75. Well done!

I went back analyzing the reranker performances but as soon as I started,  another detail got my attention: those contextualizations were so strange...

## Interlude: what do I want Minerva to become?

During the July pause, I read [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (Gao et al., 2023) and started thinking about how many things I would have to consider adding to Minerva in the future.

That survey describes many advanced techniques, such as query rewriting and multi-hop retrieval (following leads from one document to another). They are needed when the RAG system is autonomous and runs its own pipeline. In that scenario, they have every reason to be there.

Then I realized something. I had used Minerva through its MCP server for months without drawing the obvious conclusion: the agent using Minerva already provides many of these techniques. It rewrites the query, searches again and follows leads across documents by itself. I did not need to build them into Minerva.

I'm not the only guy thinking this is wise: PostHog built their own AI agent and [wished they had started with an MCP server](https://newsletter.posthog.com/p/what-we-wish-we-knew-before-building).

So, what is Minerva? It is a local RAG system meant to be used through an MCP server. It provides a high-quality base search and other composable tools, like full document retrieval. Deciding the pipeline is the job of the AI application that uses it: something like "your-favourite-foundation-model-here Desktop", or a thinking model running locally in LM Studio.

## Contextualization wars

It was late August 2026. Looking at the contextualization records, I found that Qwen's were complete crap. In the best cases they were wrong. In the worst, they were hallucinations: Qwen invented facts and repeated words over and over. Of over 134k prefixes (the contexts added in front of each chunk), almost 2,700 were longer than 2,000 characters and about 200 were longer than 40,000.

So, let me recap:

- the fast model produced garbage
- the slow model took ages, and still no improvement was visible

I attacked both problems. First, I looked for a better small model. After a few tests on wp111, Gemma 4 E2B (a lighter version of the Gemma model I had used so far) and Qwen3-1.7B had the same MRR@10. I picked E2B because its contexts were shorter and did not repeat the section heading text (Qwen had this "header echo" in 16.5% of the samples).

Then I cut ingestion time with two very effective moves:

- smaller corpus: half of the text of the Wikipedia articles was in the references section, so I removed it
- thinking off: LM Studio logs showed that most inference time was spent thinking. Turning it off was a great move: with E2B, ingesting the whole corpus took about 50 hours.

Faster ingestion finally let me experiment with contextualization itself. I tried several strategies on wp111 first: a header breadcrumb (the chain of section titles above the chunk), the same context for all chunks of a document, alternative prompts, and a more complex summary. The complex summary was my attempt to get around the compromise I described earlier. A small model cannot read a whole document, so I built a document summary from the summaries of its segments. The model then received both this and the summary of the chunk's own segment.

None of them helped, and some made results worse. Giving the same context to all chunks of a document added the same topic words to each chunk, and made them all look alike.

The most promising strategy, the complex summary, went on to the full corpus. These are the results on wp1283 at α = 0.7, with the reranker off:

| contextualization | R@10 | MRR@10 | worst gold rank |
| --- | --- | --- | --- |
| none | 0.984 | 0.794 | 13 |
| segment summary only (the original one) | 0.964 | 0.796 | 20 |
| document + segment summary | 0.966 | 0.800 | 31 |

The complex version did not improve on the original one: the difference is noise. Both were also slightly worse than no contextualization at all. R@10 and MRR@10 were about the same, but the worst gold source fell from rank 13 to rank 20 and rank 31.

When I looked more deeply at the contexts, I found the real problem: all my attempts were heading in the wrong direction. Even Gemma wrote a _summary_, not a context. A good context does not summarize the document. _It adds what the chunk needs but does not say: the subject when the chunk does not name it, and the references it leaves unresolved._ Examples are "Michael" meaning George Michael, "this cat" and "the previous quarter". This is also where my early compromise came back. A summary cannot resolve "this cat", because the cat's name is in the full document, not in its summary.

In a desperate attempt to find out what a good context for a chunk would be like, I made one last try with heavier models, using the same prompt as the Anthropic article. I got a surprise: Claude Haiku, Claude Sonnet and Gemma 4 26B (the bulkiest local model I could load on my Mac) wrote summaries; only Gemini 3.1 Pro wrote a true context. But please note, this was a single test.

At this point I understood that the extra compute time was not worth the small gain in search, and I dropped contextualization.

## September victories

Once I had decided that contextualization was just a dream, I focused on my last items in the roadmap to publish the project and start using it.

Looking for an easy win, I built the MCP server for Minerva2.

Then I had to face the other problem: why was the reranker not giving the advantage it promised?

It turned out that it was, again, a problem of model size. The small reranker, bge-reranker-v2-m3, gave unsatisfying results compared to no reranker: MRR@10 rose from 0.794 to 0.841, but some gold sources fell badly, for example from rank 4 to 49, or from 2 to 15. A big, fat reranker, Qwen3-Reranker-4B (4 billion parameters), improved Minerva's performance significantly. Unfortunately, it was super slow.

Reranking 20 candidates took 1.2 s with bge-reranker-v2-m3 and 12.1 s with Qwen3-Reranker-4B. But gold sources could fall as low as rank 49, so I needed to rerank more candidates, approximately 100.

So I decided to cascade them. The small reranker pushed results up enough that every gold source was within the first 20 ranks. Then I only had to send those 20 to the big reranker, and latency became acceptable. With the cascade (bge on the top 100, then Qwen on the top 20), MRR@10 was 0.918, the worst gold rank was 5, and a query took 16.6 s.

It was clear to me that many things I read about RAG systems may work, but only with big models and unlimited tokens (even using Haiku for contextualization, ingesting wp1283 would cost hundreds of dollars). The core difficulty of Minerva2 was doing everything with local, low-performance models. It affected so many things: contextualization cost, model choice, the cascade.

At this point, the most significant features I wanted for Minerva were done. I used my evaluation pipeline to compare with version 1 and saw great improvements: the R@5 miss rate (the share of queries with no gold source in the top 5) was 7.7% for v1 and 2.4% for v2.

## Almost easy install

Finally, before publishing, I started to think about the setup needed for running Minerva:

- PostgreSQL with two extensions (pgvector and ParadeDB pg_search)
- llama.cpp running the two rerankers
- LM Studio running the document embedder
- a .NET SDK to build the binaries

Well, not exactly "easy". So I made a single move to make it at least acceptable. I switched to [llama-swap](https://github.com/mostlygeek/llama-swap) (a proxy that starts and stops local model servers on demand) for both rerankers and the document embedder, and added instructions on how to register it as a service. This meant that LM Studio was no longer needed.

I also improved configuration file management a little, deciding to adopt the [XDG](https://specifications.freedesktop.org/basedir/latest/) convention of storing them all in `~/.config/minerva/` and naming each one after its binary, while previously they all had the Microsoft standard `appsettings.json` name.

So, at least, once the installation is done, one doesn't have to open terminals and start launching CLI tools: all the needed infrastructure to run the Minerva MCP server is a service starting at login time, and everything unloads after a timeout that I set at 15 minutes, freeing memory.

At the end of September, I uninstalled legacy Minerva, installed the new one on my Mac and started using it.

At last, on 2026-10-01, I [published the project to GitHub](https://github.com/michele-orsini-71/minerva2).


## What's next?

Well, now I have to focus on other topics but sooner or later I will continue the work on this project. It was a great ride and I don't think the project is complete: there are some improvements that need to be done, others that would be nice to have.

What I think I NEED to do for a first, decent Minerva version:

1. a lighter setup: these moves, for example, would leave Homebrew as the only prerequisite on a Mac
	- SQLite + sqlite-vec and FTS5 replace PostgreSQL; there won't be HNSW but for a personal archive of a few tens of thousands of chunks the linear search is still feasible
	- ship signed and notarized binaries, so users need no .NET SDK. Publishing them self-contained would also drop the .NET Runtime dependency.
	- automate the model-server install with a script
2. add metadata search, filters (path, folder, date, tags) and a recency boost
3. a UI for easier admin, no more fiddling with json files

What else would be nice to have:

- extract and index conversations from Claude Code, Claude Desktop or any other AI chat for quick retrieval. I think that discussions with foundation models include lots of information one wants to search and consult from time to time
- consider other corpora for eval, e.g. [BEIR](https://github.com/beir-cellar/beir); I already did some preliminary work on it
- contextualization: have we finished? I don't accept that it failed, I want to give it another try because in the meantime I had some other ideas that I want to test
- eval metrics: explore more to have other insights on RAG performance for example [Discounted cumulative gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- more realistic queries in the evals: I could log the queries Claude actually sends and build a second dataset from them 
- multimodality: only markdown? why not use something like [MinerU](https://github.com/opendatalab/MinerU), which converts PDFs and other documents to markdown?
- automatic re-index when notes change
- adding observability


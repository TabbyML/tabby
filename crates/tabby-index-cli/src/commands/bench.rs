use std::{
    fs::File,
    io,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use clap::Args;
use tantivy::{
    collector::{Count, TopDocs},
    query::QueryParser,
    schema::{Field, Schema},
    Index, TantivyDocument,
};

use crate::timer::TimerTree;

#[derive(Args)]
pub struct BenchArgs {
    /// Path to the queries file
    #[clap(short, long)]
    queries: PathBuf,

    /// Number of times to repeat the benchmark
    #[clap(short, long, default_value = "1")]
    num_repeat: usize,
}

pub fn run_bench_cli(index_path: &Path, args: &BenchArgs) -> Result<(), String> {
    run_bench(index_path, &args.queries, args.num_repeat)
}

fn extract_search_fields(schema: &Schema) -> Vec<Field> {
    schema
        .fields()
        .filter(|&(_, field_entry)| field_entry.is_indexed())
        .map(|(field, _)| field)
        .collect()
}

fn read_query_file(query_path: &Path) -> io::Result<Vec<String>> {
    let query_file: File = File::open(query_path)?;
    let file = BufReader::new(&query_file);
    let mut queries = Vec::new();
    for line_res in file.lines() {
        queries.push(line_res?);
    }
    Ok(queries)
}

fn run_bench(index_path: &Path, query_filepath: &Path, num_repeat: usize) -> Result<(), String> {
    println!("Index : {index_path:?}");
    println!("Query : {query_filepath:?}");
    println!("-------------------------------\n\n\n");

    let index =
        Index::open_in_dir(index_path).map_err(|e| format!("Failed to open index.\n{e:?}"))?;
    let searcher = index.reader().map_err(|err| format!("{err:?}"))?.searcher();
    let default_search_fields: Vec<Field> = extract_search_fields(&index.schema());
    let queries = read_query_file(query_filepath)
        .map_err(|e| format!("Failed reading the query file:  {e}"))?;
    let query_parser = QueryParser::new(
        index.schema(),
        default_search_fields,
        index.tokenizers().clone(),
    );

    println!("SEARCH\n");
    println!("query\tnum hits\ttime in microsecs");
    for _ in 0..num_repeat {
        for query_txt in &queries {
            let query = query_parser
                .parse_query(query_txt)
                .unwrap_or_else(|x| panic!("Failed to parse query {query_txt:?}.\n\n{x:?}"));
            let mut timing = TimerTree::default();
            let (_top_docs, count) = {
                let _search = timing.open("search");
                searcher
                    .search(&query, &(TopDocs::with_limit(10), Count))
                    .map_err(|e| format!("Failed while searching query {query_txt:?}.\n\n{e:?}"))?
            };
            println!("{}\t{}\t{}", query_txt, count, timing.total_time());
        }
    }

    println!("\n\nFETCH STORE\n");
    println!("query\ttime in microsecs");
    for _ in 0..num_repeat {
        for query_txt in &queries {
            let query = query_parser.parse_query(query_txt).unwrap();
            let top_docs = searcher
                .search(&*query, &TopDocs::with_limit(10))
                .map_err(|e| {
                    format!("Failed while retrieving document for query {query:?}.\n{e:?}")
                })?;
            let mut timer = TimerTree::default();
            {
                let _scoped_timer_ = timer.open("total");
                for (_score, doc_address) in top_docs {
                    searcher.doc::<TantivyDocument>(doc_address).unwrap();
                }
            }
            println!("{}\t{}", query_txt, timer.total_time());
        }
    }

    Ok(())
}

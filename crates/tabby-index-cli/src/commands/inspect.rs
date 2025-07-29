use std::path::Path;

use tantivy::{self, schema::Schema, space_usage::PerFieldSpaceUsage, Index};

pub fn run_inspect_cli(directory: &Path) -> tantivy::Result<()> {
    let index = Index::open_in_dir(directory)?;
    let schema = index.schema();
    let searcher = index.reader()?.searcher();
    let segments = searcher.segment_readers();
    println!();
    println!("===============================================================================");
    println!("Inspect index report");
    println!("===============================================================================\n");
    println!("1. General infos");
    println!("===============================================================================");
    println!("Index directory: {directory:?}");
    println!("Number of segments: {}", segments.len());
    let space_usage = searcher.space_usage()?;
    println!("Total bytes: {}", space_usage.total());
    println!();
    for (i, (segment_reader, segment_space_usage)) in segments
        .iter()
        .zip(space_usage.segments().iter())
        .enumerate()
    {
        let section_count = i + 2;
        println!(
            "{}. Space usage for segment: `{}`",
            section_count,
            segment_reader.segment_id().uuid_string()
        );
        println!("==============================================================================");
        println!("Num docs: {}", segment_space_usage.num_docs());
        println!("Store space usage:");
        println!("Total bytes: {}", segment_space_usage.store().total());
        println!(
            "Offset bytes: {}",
            segment_space_usage.store().offsets_usage()
        );
        println!();
        println!("{section_count}.1 Term dictionnary space usage");
        println!("--------------------------------");
        let per_field_space_usage = segment_space_usage.termdict();
        println!("Total bytes: {}", per_field_space_usage.total());
        print_fields_space_usage(&schema, per_field_space_usage);
        println!();

        println!("{section_count}.2 Fast field space usage");
        println!("--------------------------------");
        let fast_field_space_usage = segment_space_usage.fast_fields();
        println!("Total bytes: {}", fast_field_space_usage.total());
        print_fields_space_usage(&schema, fast_field_space_usage);
        println!();

        println!("{section_count}.3 Postings space usage");
        println!("--------------------------------");
        let postings_space_usage = segment_space_usage.postings();
        println!("Total bytes: {}", postings_space_usage.total());
        print_fields_space_usage(&schema, postings_space_usage);
        println!();

        println!("{section_count}.4 Positions space usage");
        println!("--------------------------------");
        let positions_space_usage = segment_space_usage.positions();
        println!("Total bytes: {}", positions_space_usage.total());
        print_fields_space_usage(&schema, positions_space_usage);
        println!();
    }

    println!();
    println!("----------------------------- END OF REPORT ----------------------------------");
    Ok(())
}

fn print_fields_space_usage(schema: &Schema, per_field_space_usage: &PerFieldSpaceUsage) {
    println!("Total bytes: {}", per_field_space_usage.total());
    for (field, field_space_usage) in per_field_space_usage.fields() {
        let field_name = schema.get_field_name(*field);
        println!(
            "Field `{}` bytes: {}",
            field_name,
            field_space_usage.total()
        );
    }
}

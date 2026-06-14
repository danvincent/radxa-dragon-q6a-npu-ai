pub struct TextSplitter {
    chunk_size: usize,
    #[allow(dead_code)]
    chunk_overlap: usize,
}

impl TextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self { chunk_size, chunk_overlap }
    }

    pub fn split(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .chunks(self.chunk_size)
            .map(|chunk| chunk.join(" "))
            .collect()
    }
}

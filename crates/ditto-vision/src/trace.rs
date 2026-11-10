//! Flat pictures as SVG: screenshots, diagrams, logos, icons.
//!
//! The picture is reduced to a small palette, and every palette colour becomes one path
//! that follows the pixel edges of its area. Rendering the SVG gives back the reduced
//! picture **exactly**, pixel for pixel, at any zoom: there is nothing to tune and nothing
//! that can come out subtly wrong, which is what makes screenshot-to-SVG tools frustrating.
//! Holes are handled by the even-odd fill rule, so a ring is one path.
//!
//! This is for pictures made of flat areas. A photograph can be traced, but the result is
//! a large SVG of a posterised photo; [`Traced::bytes_per_pixel`] tells the caller when
//! that happened.

use image::RgbaImage;
use std::collections::HashMap;
use std::fmt::Write;

/// A pixel edge from one corner to the next.
type Edge = ((u32, u32), (u32, u32));

/// One palette colour and the outlines of its area: closed loops of corner points, in
/// pixel-corner coordinates (`0..=width`, `0..=height`).
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub rgb: [u8; 3],
    pub pixels: usize,
    pub loops: Vec<Vec<(u32, u32)>>,
}

#[derive(Debug, Clone)]
pub struct Traced {
    pub width: u32,
    pub height: u32,
    /// Largest area first. Transparent pixels belong to no layer.
    pub layers: Vec<Layer>,
    /// The colours the picture was reduced to.
    pub palette: Vec<[u8; 3]>,
    /// The index into `palette` of every pixel, `None` where transparent. What the SVG
    /// reproduces.
    pub reduced: Vec<Option<u8>>,
    /// Distinct opaque colours in the original (counted up to 65,536).
    pub original_colours: usize,
}

/// Reduce `img` to at most `max_colours` (2 to 256) and outline every colour.
pub fn trace(img: &RgbaImage, max_colours: usize) -> Traced {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let max_colours = max_colours.clamp(2, 256);
    let (palette, reduced, original_colours) = reduce(img, max_colours);

    // Every pixel edge between a colour and anything else, directed so the colour is on
    // the right-hand side when walking along it (clockwise on screen).
    let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); palette.len()];
    let at = |x: isize, y: isize| -> Option<u8> {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            None
        } else {
            reduced[y as usize * w + x as usize]
        }
    };
    let mut counts = vec![0usize; palette.len()];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let Some(c) = at(x, y) else { continue };
            counts[c as usize] += 1;
            let (x0, y0, x1, y1) = (x as u32, y as u32, x as u32 + 1, y as u32 + 1);
            let e = &mut edges[c as usize];
            if at(x, y - 1) != Some(c) {
                e.push(((x0, y0), (x1, y0)));
            }
            if at(x + 1, y) != Some(c) {
                e.push(((x1, y0), (x1, y1)));
            }
            if at(x, y + 1) != Some(c) {
                e.push(((x1, y1), (x0, y1)));
            }
            if at(x - 1, y) != Some(c) {
                e.push(((x0, y1), (x0, y0)));
            }
        }
    }

    let mut layers: Vec<Layer> = palette
        .iter()
        .zip(edges)
        .zip(counts)
        .filter(|(_, n)| *n > 0)
        .map(|((rgb, e), pixels)| Layer {
            rgb: *rgb,
            pixels,
            loops: link(e),
        })
        .collect();
    layers.sort_by(|a, b| b.pixels.cmp(&a.pixels).then(a.rgb.cmp(&b.rgb)));
    Traced {
        width: w as u32,
        height: h as u32,
        layers,
        palette,
        reduced,
        original_colours,
    }
}

/// Join directed edges into closed loops and drop the corners that lie on a straight line.
fn link(edges: Vec<Edge>) -> Vec<Vec<(u32, u32)>> {
    // edges leaving each corner: one, or two where two areas of the colour touch diagonally
    let mut from: HashMap<(u32, u32), [usize; 2]> = HashMap::with_capacity(edges.len());
    const NONE: usize = usize::MAX;
    for (i, (a, _)) in edges.iter().enumerate() {
        let slot = from.entry(*a).or_insert([NONE, NONE]);
        slot[if slot[0] == NONE { 0 } else { 1 }] = i;
    }
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();
    for start in 0..edges.len() {
        if used[start] {
            continue;
        }
        let mut pts: Vec<(u32, u32)> = Vec::new();
        let mut i = start;
        loop {
            used[i] = true;
            let (a, b) = edges[i];
            pts.push(a);
            if b == edges[start].0 {
                break;
            }
            let out = from[&b];
            // At a diagonal touch take the right turn, which keeps the two areas apart.
            // Every corner has as many edges in as out, so an unused one always exists.
            let dir = (b.0 as i64 - a.0 as i64, b.1 as i64 - a.1 as i64);
            let right = (-dir.1, dir.0);
            let pick = |j: usize| j != NONE && !used[j];
            let turn = |j: usize| {
                (
                    edges[j].1 .0 as i64 - b.0 as i64,
                    edges[j].1 .1 as i64 - b.1 as i64,
                ) == right
            };
            i = match (pick(out[0]), pick(out[1])) {
                (true, true) => {
                    if turn(out[0]) {
                        out[0]
                    } else {
                        out[1]
                    }
                }
                (true, false) => out[0],
                (false, true) => out[1],
                (false, false) => unreachable!("an outline that does not close"),
            };
        }
        // keep only the corners where the direction changes
        let n = pts.len();
        let corners: Vec<(u32, u32)> = (0..n)
            .filter(|&k| {
                let (p, c, q) = (pts[(k + n - 1) % n], pts[k], pts[(k + 1) % n]);
                (p.0 == c.0) != (c.0 == q.0)
            })
            .map(|k| pts[k])
            .collect();
        loops.push(corners);
    }
    loops
}

/// Palette, palette index per pixel, and the number of distinct opaque colours.
fn reduce(img: &RgbaImage, max_colours: usize) -> (Vec<[u8; 3]>, Vec<Option<u8>>, usize) {
    let mut hist: HashMap<[u8; 3], u32> = HashMap::new();
    for p in img.pixels() {
        if p.0[3] >= 128 && (hist.len() < 65_536 || hist.contains_key(&[p.0[0], p.0[1], p.0[2]])) {
            *hist.entry([p.0[0], p.0[1], p.0[2]]).or_insert(0) += 1;
        }
    }
    let distinct = hist.len();
    let mut colours: Vec<([u8; 3], u32)> = hist.into_iter().collect();
    colours.sort(); // the map's order is random; the result must not be

    let palette: Vec<[u8; 3]> = if colours.len() <= max_colours {
        colours.iter().map(|c| c.0).collect()
    } else {
        median_cut(colours, max_colours)
    };
    let mut cache: HashMap<[u8; 3], u8> = HashMap::new();
    let reduced = img
        .pixels()
        .map(|p| {
            if p.0[3] < 128 {
                return None;
            }
            let rgb = [p.0[0], p.0[1], p.0[2]];
            Some(*cache.entry(rgb).or_insert_with(|| {
                let d = |c: &[u8; 3]| {
                    (0..3)
                        .map(|k| (c[k] as i32 - rgb[k] as i32).pow(2))
                        .sum::<i32>()
                };
                palette
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, c)| d(c))
                    .map_or(0, |(i, _)| i as u8)
            }))
        })
        .collect();
    (palette, reduced, distinct)
}

/// Split the colours into `n` boxes, always cutting the box with the widest spread at its
/// weighted middle; each box becomes its mean colour.
fn median_cut(colours: Vec<([u8; 3], u32)>, n: usize) -> Vec<[u8; 3]> {
    let spread = |b: &[([u8; 3], u32)]| -> (usize, u32) {
        (0..3)
            .map(|k| {
                let (lo, hi) = b
                    .iter()
                    .fold((255u8, 0u8), |(lo, hi), c| (lo.min(c.0[k]), hi.max(c.0[k])));
                (k, (hi - lo) as u32)
            })
            .max_by_key(|x| x.1)
            .unwrap()
    };
    let mut boxes = vec![colours];
    while boxes.len() < n {
        // the box whose widest channel, weighted by how many pixels it holds, is largest
        let Some((bi, (ch, _))) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| (i, spread(b)))
            .filter(|(_, s)| s.1 > 0)
            .max_by_key(|(i, s)| s.1 as u64 * boxes[*i].iter().map(|c| c.1 as u64).sum::<u64>())
        else {
            break;
        };
        let mut b = boxes.swap_remove(bi);
        b.sort_by_key(|c| (c.0[ch], c.0));
        let total: u64 = b.iter().map(|c| c.1 as u64).sum();
        let (mut acc, mut cut) = (0u64, 1);
        for (i, c) in b.iter().enumerate() {
            acc += c.1 as u64;
            if acc * 2 >= total {
                cut = (i + 1).clamp(1, b.len() - 1);
                break;
            }
        }
        let hi = b.split_off(cut);
        boxes.push(b);
        boxes.push(hi);
    }
    let mut out: Vec<[u8; 3]> = boxes
        .iter()
        .map(|b| {
            let total: u64 = b.iter().map(|c| c.1 as u64).sum();
            let mut m = [0u8; 3];
            for k in 0..3 {
                m[k] = ((b.iter().map(|c| c.0[k] as u64 * c.1 as u64).sum::<u64>() + total / 2)
                    / total) as u8;
            }
            m
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

impl Traced {
    /// A complete SVG document.
    pub fn to_svg(&self) -> String {
        let (w, h) = (self.width, self.height);
        let mut s = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" shape-rendering=\"crispEdges\">\n"
        );
        let opaque = self.reduced.iter().all(Option::is_some);
        for (i, l) in self.layers.iter().enumerate() {
            let fill = format!("#{:02x}{:02x}{:02x}", l.rgb[0], l.rgb[1], l.rgb[2]);
            // With no transparency the largest area is simply the backdrop: everything
            // else is drawn on top of it and covers exactly what is not its own.
            if i == 0 && opaque {
                let _ = writeln!(s, "<rect width=\"{w}\" height=\"{h}\" fill=\"{fill}\"/>");
                continue;
            }
            let _ = write!(s, "<path fill=\"{fill}\" fill-rule=\"evenodd\" d=\"");
            for lp in &l.loops {
                let _ = write!(s, "M{} {}", lp[0].0, lp[0].1);
                for k in 1..lp.len() {
                    // outlines alternate between horizontal and vertical moves
                    if lp[k].1 == lp[k - 1].1 {
                        let _ = write!(s, "H{}", lp[k].0);
                    } else {
                        let _ = write!(s, "V{}", lp[k].1);
                    }
                }
                s.push('Z');
            }
            s.push_str("\"/>\n");
        }
        s.push_str("</svg>\n");
        s
    }

    /// The reduced picture as pixels: what a correct renderer must show for the SVG.
    pub fn to_image(&self) -> RgbaImage {
        RgbaImage::from_fn(self.width, self.height, |x, y| {
            match self.reduced[(y * self.width + x) as usize] {
                Some(i) => {
                    let c = self.palette[i as usize];
                    image::Rgba([c[0], c[1], c[2], 255])
                }
                None => image::Rgba([0, 0, 0, 0]),
            }
        })
    }

    /// Fill the outlines by the even-odd rule, the way an SVG renderer does, sampling the
    /// centre of every pixel. Independent of how the outlines were made: used to prove
    /// that they reproduce the picture.
    pub fn rasterise(&self) -> Vec<Option<[u8; 3]>> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut out = vec![None; w * h];
        for l in &self.layers {
            // vertical edges crossing each row's centre line, by x
            let mut crossings: Vec<Vec<u32>> = vec![Vec::new(); h];
            for lp in &l.loops {
                for k in 0..lp.len() {
                    let (a, b) = (lp[k], lp[(k + 1) % lp.len()]);
                    if a.0 == b.0 {
                        for row in crossings
                            .iter_mut()
                            .take(a.1.max(b.1) as usize)
                            .skip(a.1.min(b.1) as usize)
                        {
                            row.push(a.0);
                        }
                    }
                }
            }
            for (y, xs) in crossings.iter_mut().enumerate() {
                xs.sort_unstable();
                for pair in xs.chunks(2) {
                    for x in pair[0]..pair[1] {
                        out[y * w + x as usize] = Some(l.rgb);
                    }
                }
            }
        }
        out
    }

    /// SVG bytes per pixel of the picture. Flat pictures are far below 0.1; a photograph
    /// is around 1 or more, larger than the PNG it came from.
    pub fn bytes_per_pixel(&self, svg_len: usize) -> f64 {
        svg_len as f64 / (self.width as f64 * self.height as f64)
    }

    /// Corner points in all outlines: a measure of how complicated the result is.
    pub fn points(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| &l.loops)
            .map(Vec::len)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::noise;
    use image::Rgba;

    const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);
    const RED: Rgba<u8> = Rgba([200, 30, 30, 255]);
    const BLUE: Rgba<u8> = Rgba([20, 60, 200, 255]);
    const CLEAR: Rgba<u8> = Rgba([0, 0, 0, 0]);

    /// The outlines, filled as a renderer fills them, must give back the reduced picture.
    fn assert_exact(img: &RgbaImage, colours: usize) -> Traced {
        let t = trace(img, colours);
        let want = t.to_image();
        let got = t.rasterise();
        for (i, p) in want.pixels().enumerate() {
            let w = (p.0[3] > 0).then_some([p.0[0], p.0[1], p.0[2]]);
            assert_eq!(got[i], w, "pixel {} of {}x{}", i, t.width, t.height);
        }
        for l in &t.layers {
            for lp in &l.loops {
                assert!(
                    lp.len() >= 4 && lp.len() % 2 == 0,
                    "an outline of {} corners",
                    lp.len()
                );
            }
        }
        t
    }

    #[test]
    fn a_rectangle_is_four_corners() {
        let mut img = RgbaImage::from_pixel(10, 8, WHITE);
        for x in 2..7 {
            for y in 1..4 {
                img.put_pixel(x, y, RED);
            }
        }
        let t = assert_exact(&img, 16);
        assert_eq!(t.layers.len(), 2);
        assert_eq!(
            t.layers[1].loops,
            vec![vec![(2, 1), (7, 1), (7, 4), (2, 4)]]
        );
        assert_eq!(
            t.to_svg(),
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"8\" viewBox=\"0 0 10 8\" shape-rendering=\"crispEdges\">\n\
             <rect width=\"10\" height=\"8\" fill=\"#ffffff\"/>\n\
             <path fill=\"#c81e1e\" fill-rule=\"evenodd\" d=\"M2 1H7V4H2Z\"/>\n\
             </svg>\n"
        );
    }

    #[test]
    fn holes_islands_and_diagonal_touches() {
        // a ring with an island in its hole
        let mut img = RgbaImage::from_pixel(12, 12, WHITE);
        for x in 1..11 {
            for y in 1..11 {
                let ring = !(3..9).contains(&x) || !(3..9).contains(&y);
                let island = (5..7).contains(&x) && (5..7).contains(&y);
                if ring || island {
                    img.put_pixel(x, y, BLUE);
                }
            }
        }
        let t = assert_exact(&img, 16);
        let blue = t.layers.iter().find(|l| l.rgb == [20, 60, 200]).unwrap();
        assert_eq!(blue.loops.len(), 3); // outside, hole, island

        // a checkerboard: every square touches its neighbours only at corners
        let check = RgbaImage::from_fn(6, 6, |x, y| if (x + y) % 2 == 0 { RED } else { BLUE });
        let t = assert_exact(&check, 16);
        assert_eq!(t.layers.iter().map(|l| l.loops.len()).sum::<usize>(), 36);

        // a diagonal line and an L shape
        let mut img = RgbaImage::from_pixel(9, 9, WHITE);
        for i in 0..9 {
            img.put_pixel(i, i, RED);
            img.put_pixel(0, i, BLUE);
            img.put_pixel(i, 8, BLUE);
        }
        assert_exact(&img, 16);
    }

    #[test]
    fn transparency_is_kept() {
        let img = RgbaImage::from_fn(8, 8, |x, y| if x < 4 && y < 4 { RED } else { CLEAR });
        let t = assert_exact(&img, 16);
        assert_eq!(t.layers.len(), 1);
        let svg = t.to_svg();
        assert!(
            !svg.contains("<rect"),
            "a transparent picture has no backdrop"
        );
        assert!(svg.contains("M0 0H4V4H0Z"));
        // all clear: an empty but valid document
        let t = assert_exact(&RgbaImage::from_pixel(3, 3, CLEAR), 16);
        assert!(t.layers.is_empty());
        assert!(t.to_svg().ends_with("</svg>\n"));
    }

    #[test]
    fn many_colours_are_reduced_and_the_result_is_still_exact() {
        let n = noise(40 * 30 * 3, 7);
        let img = RgbaImage::from_fn(40, 30, |x, y| {
            let i = ((y * 40 + x) * 3) as usize;
            Rgba([n[i], n[i + 1], n[i + 2], 255])
        });
        let t = assert_exact(&img, 8);
        assert!(t.layers.len() <= 8 && t.layers.len() >= 4);
        assert_eq!(t.original_colours, 1200);
        // noise is the worst case: the SVG is large, and the caller can see that
        assert!(t.bytes_per_pixel(t.to_svg().len()) > 1.0);

        // a gradient goes to the nearest of the palette, in bands
        let grad = RgbaImage::from_fn(64, 4, |x, _| {
            Rgba([(x * 4) as u8, (x * 4) as u8, (x * 4) as u8, 255])
        });
        let t = assert_exact(&grad, 4);
        assert_eq!(t.layers.len(), 4);
        assert!(t
            .layers
            .iter()
            .all(|l| l.loops.len() == 1 && l.loops[0].len() == 4));
    }

    #[test]
    fn exact_colours_are_kept_when_they_fit() {
        let img = RgbaImage::from_fn(30, 3, |x, _| [RED, BLUE, WHITE][(x / 10) as usize]);
        let t = trace(&img, 3);
        let mut rgb: Vec<[u8; 3]> = t.layers.iter().map(|l| l.rgb).collect();
        rgb.sort();
        assert_eq!(rgb, vec![[20, 60, 200], [200, 30, 30], [255, 255, 255]]);
        assert_eq!(t.points(), 12);
    }

    #[test]
    fn the_same_picture_gives_the_same_svg() {
        let n = noise(50 * 50 * 3, 11);
        let img = RgbaImage::from_fn(50, 50, |x, y| {
            let i = ((y * 50 + x) * 3) as usize;
            Rgba([n[i] & 0xC0, n[i + 1] & 0xC0, n[i + 2] & 0xC0, 255])
        });
        assert_eq!(trace(&img, 12).to_svg(), trace(&img, 12).to_svg());
    }

    #[test]
    fn a_single_pixel_and_a_single_row() {
        assert_exact(&RgbaImage::from_pixel(1, 1, RED), 2);
        assert_exact(
            &RgbaImage::from_fn(7, 1, |x, _| if x % 2 == 0 { RED } else { BLUE }),
            2,
        );
    }
}

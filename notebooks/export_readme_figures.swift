// Rasterize the existing paper PDFs for GitHub; does not rerun experiments or plots.
// Run on macOS from the repository root: swift notebooks/export_readme_figures.swift
import AppKit
import CoreGraphics
import Foundation

func render(_ page: CGPDFPage, width: Int) -> CGImage {
    let bounds = page.getBoxRect(.cropBox)
    let height = Int(ceil(Double(width) * bounds.height / bounds.width))
    guard let context = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        fatalError("Could not create a raster context")
    }
    let target = CGRect(x: 0, y: 0, width: width, height: height)
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(target)
    precondition(page.rotationAngle == 0, "Expected an unrotated paper figure")
    let scale = Double(width) / bounds.width
    context.scaleBy(x: scale, y: scale)
    context.translateBy(x: -bounds.minX, y: -bounds.minY)
    context.drawPDFPage(page)
    return context.makeImage()!
}

// Locate the artwork, including labels, rather than preserving blank PDF margins.
func contentBounds(_ image: CGImage) -> CGRect {
    let bitmap = NSBitmapImageRep(cgImage: image)
    let pixels = bitmap.bitmapData!
    var left = image.width, right = -1, top = image.height, bottom = -1
    for y in 0..<image.height {
        for x in 0..<image.width {
            let i = y * bitmap.bytesPerRow + x * bitmap.samplesPerPixel
            if pixels[i] < 250 || pixels[i + 1] < 250 || pixels[i + 2] < 250 {
                left = min(left, x); right = max(right, x)
                top = min(top, y); bottom = max(bottom, y)
            }
        }
    }
    precondition(right >= left, "PDF contains no visible artwork")
    let padding = max(8, Int(Double(right - left + 1) * 0.012))
    return CGRect(x: max(0, left - padding), y: max(0, top - padding),
                  width: min(image.width - 1, right + padding) - max(0, left - padding) + 1,
                  height: min(image.height - 1, bottom + padding) - max(0, top - padding) + 1)
}

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let output = root.appendingPathComponent("docs/figures")
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let figures = ["pareto_gsq_rco_both", "qadd_training", "bars_disag_both", "odp_timeline"]

for name in figures {
    let source = root.appendingPathComponent("notebooks/figures/\(name).pdf")
    guard let document = CGPDFDocument(source as CFURL), document.numberOfPages == 1,
          let page = document.page(at: 1) else {
        fatalError("Expected a single-page PDF: \(source.path)")
    }
    let preview = render(page, width: 1800)
    let width = Int(ceil(1800.0 * 1800.0 / contentBounds(preview).width))
    // Rasterize again at the required resolution; do not enlarge the preview bitmap.
    let full = render(page, width: width)
    let image = full.cropping(to: contentBounds(full))!
    guard let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
        fatalError("Could not encode \(name)")
    }
    let destination = output.appendingPathComponent("\(name).png")
    try png.write(to: destination)
    print("\(destination.lastPathComponent): \(image.width) × \(image.height), \(png.count / 1024) KiB")
}
